import sys
sys.path.append('/workspace/gan/3D-FM-GAN/model')
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends
from torchvision.utils import save_image
import numpy as np
from tqdm.auto import tqdm

from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler



from model.networks import *
from model.dataset import *
import metrics.fid as fid
from metrics.extractor import Extractor
import metrics.features as features
import utils
import wandb
import gc

from datetime import timedelta
import logging


from model.losses import IDLoss, ContentLoss, g_hinge, g_nonsaturating_loss, d_hinge, d_logistic_loss

gan_loss_list = {'hinge' : [g_hinge, d_hinge],
                'basic' : [g_nonsaturating_loss, d_logistic_loss]}



def exists(x):
    return x is not None

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()




def train_one_step(fmgan, recon_dl, disen_dl, d_optimizer, g_optimizer, device):
    """
    generator : ddp
    """
    requires_grad(fmgan.generator, True)
    
    data = next(recon_dl)
    data['p1'], data['p2'] = data['p1'].to(device, non_blocking=True), data['r1'].to(device, non_blocking=True)

    requires_grad(fmgan.discriminator, True)
    p1_hat = fmgan.generator(data['p1'], data['r1'])
    fake_logit = fmgan.discriminator(p1_hat.detach())
    real_logit = fmgan.discriminator(data['p1'])
    d_recon_loss = fmgan.d_loss(real_logit, fake_logit)

    d_recon_loss.backward()
    # accelerator.clip_grad_norm_(fmgan.discriminator.parameters(), 1.0)
    d_optimizer.step()
    d_optimizer.zero_grad(set_to_none=True)


    requires_grad(fmgan.discriminator, False)
    fake_logit = fmgan.discriminator(p1_hat)
    g_recon_loss =  fmgan.g_loss(fake_logit)
    g_recon_loss +=  fmgan.id_lambda * fmgan.id_loss(data['p1'], p1_hat).mean()
    g_recon_loss += fmgan.l1_lambda * fmgan.l1_loss(data['p1'], p1_hat)
    g_recon_loss += fmgan.lpips_lambda * fmgan.lpips_loss(data['p1'], p1_hat).mean()
    
    g_recon_loss.backward()
    # accelerator.clip_grad_norm_(fmgan.generator.parameters(), 1.0)
    g_optimizer.step()
    g_optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    # requires_grad(accelerator.unwrap_model(fmgan.discriminator), True)
    
    # disentangle
    first_data, second_data = next(disen_dl)
    first_data['p_img'], first_data['r_img'], first_data['m_img'], second_data['p_img'], second_data['r_img'], second_data['m_img'] = first_data['p_img'].to(device, non_blocking=True), first_data['r_img'].to(device, non_blocking=True), first_data['m_img'].to(device, non_blocking=True), second_data['p_img'].to(device, non_blocking=True), second_data['r_img'].to(device, non_blocking=True), second_data['m_img'].to(device, non_blocking=True)


    # D First
    requires_grad(fmgan.discriminator, True)
    p1_hat = fmgan.generator(first_data['p_img'], second_data['r_img'])
    p1_target = second_data['p_img']
    p1_fake_logit = fmgan.discriminator(p1_hat.detach())
    p1_real_logit = fmgan.discriminator(p1_target)
    d_disen_loss_1 =  fmgan.d_loss(p1_real_logit, p1_fake_logit)

    d_disen_loss_1.backward()
    
    p2_hat = fmgan.generator(second_data['p_img'], first_data['r_img'])
    p2_target = first_data['p_img']
    p2_fake_logit = fmgan.discriminator(p2_hat.detach())
    p2_real_logit = fmgan.discriminator(p2_target)
    d_disen_loss_2 = fmgan.d_loss(p2_real_logit, p2_fake_logit)

    d_disen_loss_2.backward()

    # accelerator.clip_grad_norm_(fmgan.discriminator.parameters(), 1.0)
    d_optimizer.step()
    d_optimizer.zero_grad(set_to_none=True)
   
    requires_grad(fmgan.discriminator, False)
    p1_fake_logit = fmgan.discriminator(p1_hat)
    g_disen_loss_1 = fmgan.g_loss(p1_fake_logit)
    g_disen_loss_1  += fmgan.id_lambda * fmgan.id_loss(p1_target , p1_hat).mean()
    g_disen_loss_1  += fmgan.l1_lambda * fmgan.l1_loss(p1_target , p1_hat)
    g_disen_loss_1  += fmgan.lpips_lambda * fmgan.lpips_loss(p1_target , p1_hat).mean()
    g_disen_loss_1  += fmgan.content_lambda * fmgan.content_loss(p1_hat, second_data['r_img'], second_data['m_img']).mean() # m2와 r2를 이용

    g_disen_loss_1.backward()

    p2_fake_logit = fmgan.discriminator(p2_hat)
    g_disen_loss_2 = fmgan.g_loss(p2_fake_logit)
    g_disen_loss_2 += fmgan.id_lambda * fmgan.id_loss(p2_target , p2_hat).mean()
    g_disen_loss_2 += fmgan.l1_lambda * fmgan.l1_loss(p2_target , p2_hat)
    g_disen_loss_2 += fmgan.lpips_lambda * fmgan.lpips_loss(p2_target , p2_hat).mean()
    g_disen_loss_2 += fmgan.content_lambda * fmgan.content_loss(p2_hat, first_data['r_img'], first_data['m_img']).mean()
            
    g_disen_loss_2.backward()
    # accelerator.clip_grad_norm_(fmgan.generator.parameters(), 1.0)
    g_optimizer.step()
    g_optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.empty_cache()
    # requires_grad(accelerator.unwrap_model(fmgan.discriminator), True)
    # accelerator.wait_for_everyone()

    return {'d_recon_loss' : d_recon_loss.item(), 'g_recon_loss' : g_recon_loss.item(), 'd_disen_loss_1' : d_disen_loss_1.item(), 'd_disen_loss_2' : d_disen_loss_2.item(), 'g_disen_loss_1' : g_disen_loss_1.item(), 'g_disen_loss_2' : g_disen_loss_2.item()}


def post_train_step(cfg, train_losses, fmgan, g_optimizer, d_optimizer, 
                    step, eval_fake_dl, logger):
    # -----------------------------------------------------------------------------
    # Logging loss
    # -----------------------------------------------------------------------------
    if (not step % cfg.log_interval) and (get_rank() == 0) and (cfg.with_tracking):
        # tracker logging
        wandb.log(train_losses, step=step)
        logger.info('%s', train_losses)
    
    # -----------------------------------------------------------------------------
    # Evaluation Metrics
    # -----------------------------------------------------------------------------
    if (step % cfg.evaluation_interval == 0) and (step != 0):
        if get_rank() == 0 : 
            logger.info(f"generate_images_and_stack_features")
        image_holder = []
        for idx, data in enumerate(eval_fake_dl):
            with torch.inference_mode():
                p1_hat = fmgan.generator(data['p1'], data['r1'])
                image_holder.append(p1_hat)
            if idx == 0:
                break
        fake_imgs = torch.cat(image_holder, 0)

        if dist.get_rank() == 0:
            logger.info(f"Image sampling")
            save_folder = cfg.exp_path.joinpath('figures')
            save_folder.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(fake_imgs[:cfg.num_generation].detach(), start=1):
                save_path = cfg.exp_path.joinpath(save_folder, f'images_{step}_{idx}.png')
                save_image(((img+1)/2).clamp(0.0, 1.0), save_path, padding=0)
                
            if (cfg.with_tracking):
                wandb.log({"generated_images": wandb.Image(fake_imgs[:cfg.num_generation])}, step=step)

    # -----------------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------------
    # TODO : best model 저장
    if (not step % cfg.ckpt_interval or not step % cfg.total_step) and (dist.get_rank()==0):
        dict_states = {'g' : fmgan.generator.module.state_dict(),
                       'd' : fmgan.discriminator.module.state_dict(),
                       'g_optimizer' : g_optimizer.state_dict(),
                       'd_optimizer' : d_optimizer.state_dict(),
                       'step' : step,
                       'seed' : cfg.seed,
                       'exp_name' : cfg.exp_name}
        torch.save(dict_states, cfg.exp_path.joinpath(f'model-{step}.pt'))

@hydra.main(config_path="config", config_name="main.yaml", version_base=None)
def main(cfg):
    # -----------------------------------------------------------------------------
    # Setup Config
    # -----------------------------------------------------------------------------
    OmegaConf.set_struct(cfg, False)
    # -----------------------------------------------------------------------------
    # Distributed Setting
    # -----------------------------------------------------------------------------
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.distributed = n_gpu > 1

    if cfg.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        init_process_group(backend='nccl', init_method="env://", timeout=timedelta(0, 18000))
        synchronize()


    # -----------------------------------------------------------------------------
    # experiment folder
    # -----------------------------------------------------------------------------
    if get_rank() == 0:
        dir_path = Path(os.path.abspath(__file__)).parent
        if cfg.exp_name is not None:
            exp_path = Path(dir_path, cfg.exp_name)
            exp_path.mkdir(parents=True, exist_ok=True)
            cfg.exp_path = exp_path
        else:
            exp_path = Path(dir_path, 'tmp')
            exp_path.mkdir(parents=True, exist_ok=True)
            cfg.exp_path = exp_path

    # -----------------------------------------------------------------------------
    # logger
    # -----------------------------------------------------------------------------
    if get_rank() == 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logging_path = exp_path.joinpath('logging')
        logging_path.mkdir(parents=True, exist_ok=True)
        cfg.logging_path = logging_path


        logger = logging.getLogger('')
        logger.addHandler(logging.FileHandler(logging_path.joinpath('run.log')))
    else:
        logger = None
    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    recon1_ds = ReconDataset(root=cfg.dataset.root,
                             img_range=cfg.dataset.recon1.range,
                             sub_dir=cfg.dataset.recon1.sub_dir,
                             crop=cfg.dataset.recon1.crop,
                             resize_size=cfg.dataset.recon1.resize,
                             random_flip=cfg.dataset.recon1.random_flip)
    recon2_ds = ReconDataset(root=cfg.dataset.root,
                             img_range=cfg.dataset.recon2.range,
                             sub_dir=cfg.dataset.recon2.sub_dir,
                             crop=cfg.dataset.recon2.crop,
                             random_flip=cfg.dataset.recon2.random_flip)
    disen_ds = DisentangleDataset(root=cfg.dataset.root,
                                  sub_dir=cfg.dataset.disen.sub_dir,
                                  crop=cfg.dataset.disen.crop,
                                  resize_size=cfg.dataset.disen.resize, 
                                  random_flip=cfg.dataset.disen.random_flip)

    eval_fake_ds = ReconDataset(root=cfg.dataset.root,
                                img_range=cfg.dataset.eval_fake.range,
                                sub_dir=cfg.dataset.eval_fake.sub_dir,
                                crop=cfg.dataset.eval_fake.crop,
                                random_flip=cfg.dataset.eval_fake.random_flip)
    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    recon1_dl = DataLoader(recon1_ds, batch_size=cfg.training.batch_size, 
                          shuffle=False, pin_memory=True, num_workers=cfg.training.loader_workers,
                          prefetch_factor=cfg.training.prefetch_factor,
                          sampler=DistributedSampler(recon1_ds))

    recon2_dl = DataLoader(recon2_ds, batch_size=cfg.training.batch_size,
                          shuffle=False, pin_memory=True, num_workers=cfg.training.loader_workers,
                          prefetch_factor=cfg.training.prefetch_factor,
                          sampler=DistributedSampler(recon2_ds))

    disen_dl = DataLoader(disen_ds, batch_size=cfg.training.batch_size,
                          shuffle=False, pin_memory=True, num_workers=cfg.training.loader_workers,
                          prefetch_factor=cfg.training.prefetch_factor,
                          sampler=DistributedSampler(disen_ds))

    eval_fake_dl = DataLoader(eval_fake_ds, batch_size=cfg.eval.batch_size, 
                        shuffle=False, pin_memory=True, num_workers=cfg.eval.loader_workers,
                        prefetch_factor=cfg.eval.prefetch_factor,
                        sampler=DistributedSampler(recon2_ds))

    recon1_dl = IterLoader(recon1_dl)
    recon2_dl = IterLoader(recon2_dl)
    disen_dl = IterLoader(disen_dl)
    # recon1_dl = sample_data(recon1_dl)
    # recon2_dl = sample_data(recon2_dl)
    # disen_dl = sample_data(disen_dl)


    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    gpu_id = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{gpu_id}")
    generator = FMGenerator(cfg.model).train().requires_grad_(True).to(gpu_id)
    discriminator = Discriminator(cfg.model.input_size).train().requires_grad_(True).to(gpu_id)

    generator = DDP(generator, device_ids=[gpu_id], broadcast_buffers=False)
    generator.requires_grad_(False)
    discriminator = DDP(discriminator, device_ids=[gpu_id], broadcast_buffers=False)
    discriminator.requires_grad_(False)
    
    # -----------------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------------
    if cfg.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.Adam8bit
    else:
        optimizer_class = torch.optim.Adam

    # TODO :optimizer learning rate from ckpt
    g_optimizer = optimizer_class(generator.parameters(),
                                  lr=cfg.optimizer.lr1, betas=(cfg.optimizer.beta1, cfg.optimizer.beta2))

    d_optimizer = optimizer_class(discriminator.parameters(),
                                  lr=cfg.optimizer.lr2, betas=(cfg.optimizer.beta1, cfg.optimizer.beta2))
    # -----------------------------------------------------------------------------
    # Losses
    # -----------------------------------------------------------------------------
    lpips_loss = LPIPS(net=cfg.training.lpips_type).eval()# .to(device)
    id_loss = IDLoss(cfg.training, 'cuda')
    l1_loss = torch.nn.L1Loss().to('cuda')
    content_loss = ContentLoss().to('cuda')
    requires_grad(lpips_loss, False)
    requires_grad(id_loss, False)

  
    if exists(ckpt.get('enc_t', None)):
        generator.enc_t.load_state_dict(ckpt['enc_t'], strict=False)
    if exists(ckpt.get('enc_w', None)):
        generator.enc_w.load_state_dict(ckpt['enc_w'], strict=False)
    if exists(ckpt.get('enc_wplus', None)):
        generator.enc_wplus.load_state_dict(ckpt['enc_wplus'], strict=False)
    if exists(ckpt.get('g', None)):
        generator.stylegan.load_state_dict(ckpt['g'], strict=False)
    if exists(ckpt.get('d', None)):
        discriminator.load_state_dict(ckpt['d'], strict=False)
    # optimizer는 따로 load할 필요가 없나?
    if exists(ckpt.get('g_optimizer', None)):
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    if exists(ckpt.get('d_optimizer', None)):
        d_optimizer.load_state_dict(ckpt['d_optimizer'])


    if get_rank() == 0:
        logger.info('config : %s', cfg)

    if cfg.training.scheme == 'from_pretrained':
        if cfg.training.from_pretrained_path is not None:
            loc = f"cuda:{gpu_id}"
            ckpt = torch.load(cfg.training.from_pretrained_path, map_location=loc)
            cfg.training.start_step= 0
            try:
                ckpt_name = os.path.basename(cfg.training.from_pretrained_path)
                # cfg.start_iter = int(os.path.splitext(ckpt_name)[0])
            except ValueError:
                pass
            seed = cfg.seed
    elif cfg.training.scheme == 'resume':
        if cfg.training.resume_path is not None:
            ckpt = torch.load(cfg.training.resume_path)
            ckpt_step = ckpt.get('step', None)
            if ckpt_step is not None:
                cfg.training.start_step = ckpt_step

            seed = ckpt.get('seed', None)
            if seed is not None:
                cfg.seed = seed
        else:
            raise RuntimeError('resume checkpoint is set to none.')
        
    elif cfg.scheme == 'scratch':
        cfg.training.start_step = 0
        ckpt = None
    else:
        raise NotImplementedError

    # -----------------------------------------------------------------------------
    # seed
    # -----------------------------------------------------------------------------
    if cfg.seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # -----------------------------------------------------------------------------
    # wandb setup
    # -----------------------------------------------------------------------------
    if get_rank() == 0 and cfg.with_tracking:
        wandb.init(project="3dfm", config=cfg)

    # -----------------------------------------------------------------------------
    # Preparation
    # -----------------------------------------------------------------------------
    fmgan = FMGAN(cfg, generator, discriminator, lpips_loss, id_loss, l1_loss, content_loss)
    if get_rank() == 0:
        logger.info(f"{'Running training':*^20}")
        logger.info(f"  reconstruction 1 num examples = {len(recon1_ds)}")
        logger.info(f"  reconstruction 2 num examples = {len(recon2_ds):*^20}")
        logger.info(f"  disentanglement num examples = {len(disen_ds):*^20}")


    gc.collect()
    torch.cuda.empty_cache()


    # -----------------------------------------------------------------------------
    # step configurations
    # -----------------------------------------------------------------------------
    if cfg.training.start_step >= cfg.training.p1_total_step:
        p1_start = cfg.training.p1_total_step
        p2_start = cfg.training.p1_total_step - cfg.training.start_step
    else:
        p1_start = cfg.training.start_step
        p2_start = 0

    p1_end = int(cfg.training.p1_total_step * 16 / (cfg.training.batch_size * n_gpu))
    p2_end = int(cfg.training.p2_total_step * 16 / (cfg.training.batch_size * n_gpu))
    # -----------------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------------
    progress_bar = tqdm(range(p1_start, p1_end), 
                        intial=p1_start, total=p1_end,
                        disable=not get_rank())
    progress_bar.set_description('Phase 1')

    for idx in progress_bar:
        step = idx + p1_start
        if step > p1_end:
            print('phase1 done')
            break
        losses = train_one_step(fmgan, recon1_dl, disen_dl, d_optimizer, g_optimizer, device)
        post_train_step(cfg, losses, fmgan, g_optimizer, d_optimizer, step, eval_fake_dl, logger)
        progress_bar.set_postfix(**losses)                
    torch.cuda.empty_cache()
  
    # -----------------------------------------------------------------------------
    # Update learning rate
    # -----------------------------------------------------------------------------
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = cfg.phase2_lr #0.001

    for param_group in d_optimizer.param_groups:
        param_group['lr'] = cfg.phase2_lr #0.001

    # -----------------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------------
    progress_bar = tqdm(range(p2_start, p2_end), 
                        intial=p2_start, total=p2_end,
                        disable=not get_rank())
    progress_bar.set_description('Phase 2')

    
    for idx in progress_bar:        
        fmgan.train()
        step = idx + p2_start
        if step > p2_end:
            print('phase2 done')
            break
        losses = train_one_step(fmgan, recon1_dl, disen_dl, d_optimizer, g_optimizer, device)
        post_train_step(cfg, losses, fmgan, g_optimizer, d_optimizer, step, eval_fake_dl, logger)
        progress_bar.set_postfix(**losses)


    # -----------------------------------------------------------------------------
    # End training
    # -----------------------------------------------------------------------------
    dist.destroy_process_group()
if __name__ == '__main__':
    main()