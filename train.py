import sys
sys.path.append('/workspace/gan/3D-FM-GAN/model')
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.backends
from torchvision.utils import save_image
import numpy as np
from tqdm.auto import tqdm

from model.networks import *
from model.dataset import *
import metrics.fid as fid
from metrics.extractor import Extractor
import metrics.features as features
import utils
import wandb

torch.backends.cudnn.benchmark = True
# Ampare GPU
# fp16을 쓰면 꺼야하는건가?
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


logger = get_logger(__name__)

def exists(x):
    return x is not None

def sample_data(loader):
  while True:
    for batch in loader:
      yield batch


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def prepare_real_moments(accelertor, data_loader, eval_model, num_generate, quantize=True):
    feature_holder = []
    for idx, data in enumerate(data_loader):
        with torch.inference_mode():
            features, logits = eval_model.get_outputs(data, quantize)
            
        feature_holder.append(features)

    feature_holder = torch.cat(feature_holder, 0)
    feature_holder = accelertor.gather_for_metrics(feature_holder)
    acts = feature_holder.detach().cpu().numpy()[:num_generate].astype(np.float64)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma
    


def train_one_step(fmgan, accelerator, recon_dl, disen_dl, d_optimizer, g_optimizer):
    # reconstruction
    # fetch data
    data = next(recon_dl)

    # training D
    requires_grad(accelerator.unwrap_model(fmgan.generator), False)
    requires_grad(accelerator.unwrap_model(fmgan.discriminator), True)
    with accelerator.autocast():
        d_recon_loss = fmgan.reconstruction_loss(data['p1'], data['r1'], 'd')

    
    accelerator.backward(d_recon_loss)
    d_optimizer.step()
    d_optimizer.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

    # training G
    requires_grad(accelerator.unwrap_model(fmgan.generator), True)
    requires_grad(accelerator.unwrap_model(fmgan.discriminator), False)
    with accelerator.autocast():
        g_recon_loss = fmgan.reconstruction_loss(data['p1'], data['r1'], 'g')

    accelerator.backward(g_recon_loss)
    g_optimizer.step()
    g_optimizer.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    
    # disentangle
    first_data, second_data = next(disen_dl)
    # training D
    requires_grad(accelerator.unwrap_model(fmgan.generator), False)
    requires_grad(accelerator.unwrap_model(fmgan.discriminator), True)
    with accelerator.autocast():
        d_disen_loss = fmgan.disentangle_loss(first_data['p_img'], first_data['r_img'], first_data['m_img'],  second_data['p_img'], second_data['r_img'], second_data['m_img'], 'd')
    accelerator.backward(d_disen_loss)
    d_optimizer.step()
    d_optimizer.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()

    # training G
    requires_grad(accelerator.unwrap_model(fmgan.generator), True)
    requires_grad(accelerator.unwrap_model(fmgan.discriminator), False)
    with accelerator.autocast():
        g_disen_loss = fmgan.disentangle_loss(first_data['p_img'], first_data['r_img'], first_data['m_img'], second_data['p_img'], second_data['r_img'], second_data['m_img'], 'g')
    accelerator.backward(g_disen_loss)
    g_optimizer.step()
    g_optimizer.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()

    return {'d_recon_loss' : d_recon_loss.item(), 'g_recon_loss' : g_recon_loss.item(), 'd_disen_loss' : d_disen_loss.item(), 'g_disen_loss' : g_disen_loss.item()}


def post_train_step(accelerator, cfg, train_losses, fmgan, eval_model, 
                    g_optimizer, d_optimizer, step, eval_fake_dl, eval_real_dl):
    # https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/loader.py
    # -----------------------------------------------------------------------------
    # Logging loss
    # -----------------------------------------------------------------------------
    if (not step % cfg.log_interval) and (accelerator.is_local_main_process) and (cfg.with_tracking):
        # tracker logging
        accelerator.log(train_losses, step=step)
        logger.info('%s', train_losses, main_process_only=True)
    
    # -----------------------------------------------------------------------------
    # Evaluation Metrics
    # -----------------------------------------------------------------------------
    if (step % cfg.evaluation_interval == 0):
        logger.info(f"Evaluation Metrics", main_process_only=True)
        logger.info(f"generate_images_and_stack_features", main_process_only=True)
        fake_feats, fake_probs, fake_imgs = features.generate_images_and_stack_features(accelerator=accelerator,
                                                                                        fmgan=fmgan,
                                                                                        eval_model=eval_model,
                                                                                        eval_fake_dl=eval_fake_dl,
                                                                                        quantize=True)
        if 'fid50k' in cfg.eval_metrics:
            logger.info(f"fid50k evaluation", main_process_only=True)
            fid_score, m1, c1 = fid.calculate_fid(accelerator, cfg, eval_model, eval_real_dl, fake_feats,
                                                  pre_cal_mean=fmgan.mean, pre_cal_std=fmgan.sigma)
            logger.info(f"{step} , fid50k = {fid_score}", main_process_only=True)
            if (accelerator.is_local_main_process) and (cfg.with_tracking):
                accelerator.log({"fid50k" : fid_score}, step=step)

        if (accelerator.is_main_process):
            logger.info(f"Image sampling", main_process_only=True)
            save_folder = cfg.exp_path.joinpath('figures')
            save_folder.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(fake_imgs[:cfg.num_generation].detach(), start=1):
                save_path = cfg.exp_path.joinpath(save_folder, 'images_{step}_{idx}.png')
                save_image(((img+1)/2).clamp(0.0, 1.0), save_path, padding=0)

            if (cfg.with_tracking):
                accelerator.log({"generated_images": wandb.Image(fake_imgs[:cfg.num_generation])}, step=step)
                
    # -----------------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------------
    # TODO : best model 저장
    if (not step % cfg.ckpt_interval or not step % cfg.total_step) and accelerator.is_main_process:
        dict_states = {'enc_t' : accelerator.get_state_dict(fmgan.generator.enc_t),
                       'enc_w' : accelerator.get_state_dict(fmgan.generator.enc_w),
                       'enc_wplus' : accelerator.get_state_dict(fmgan.generator.enc_wplus),
                       'g' : accelerator.get_state_dict(fmgan.generator.stylegan),
                       'd' : accelerator.get_state_dict(fmgan.discriminator),
                       'g_optimizer' : g_optimizer.state_dict(),
                       'd_optimizer' : d_optimizer.state_dict(),
                       'step' : step,
                       'seed' : cfg.seed,
                       'exp_name' : cfg.exp_name}

        torch.save(dict_states, cfg.exp_path.joinpath(f'model-{step}.pt'))


@hydra.main(config_path="config", config_name="main.yaml", version_base=None)
def main(cfg):
    ######### Setup Config ###############
    OmegaConf.set_struct(cfg, False)
    # primitive = OmegaConf.to_container(cfg, resolve=True)
    # print(type(primitive))
    if cfg.eval_during_training and (cfg.eval_metrics is None):
        cfg.eval_metrics = ['fid50k']

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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logging_path = exp_path.joinpath('logging')
    logging_path.mkdir(parents=True, exist_ok=True)
    logger.logger.addHandler(logging.FileHandler(logging_path.joinpath('run.log')))
    cfg.logging_path = logging_path

    # -----------------------------------------------------------------------------
    # Accelerator
    # -----------------------------------------------------------------------------
    if cfg.with_tracking:
        accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulation_steps, logging_dir=cfg.logging_path, log_with='wandb')
    else:
        accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulation_steps)
        
    logger.info(accelerator.state, main_process_only=True)
    logger.info('config : %s', cfg, main_process_only=True)
    # primitive.device = accelerator.device

    if accelerator.is_main_process and cfg.with_tracking:
        accelerator.init_trackers("3d-fmgan", cfg)
        # accelerator.init_trackers("dreambooth", config=vars(args))

    if cfg.training.scheme == 'from_pretrained':
        if cfg.training.from_pretrained_path is not None:
            ckpt = torch.load(cfg.training.from_pretrained_path)
            # torch.load(pth, map_location=device)
            try:
                ckpt_name = os.path.basename(cfg.training.from_pretrained_path)
                # cfg.start_iter = int(os.path.splitext(ckpt_name)[0])
            except ValueError:
                pass
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
        ckpt = None
    else:
        raise NotImplementedError
        
    if cfg.seed is not None:
        set_seed(cfg.seed)
        # np.random.seed(cfg.seed)
        # torch.manual_seed(cfg.seed)
        # torch.cuda.manual_seed_all(cfg.seed)
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    generator = FMGenerator(cfg.model)
    discriminator = Discriminator(cfg.model.input_size)
    
    
    if cfg.eval_during_training:
        if 'fid50k' in cfg.eval_metrics:
            extractor = Extractor(backbone=cfg.eval_backbone, post_resizer=cfg.post_resizer,
                                  device=accelerator.device)
        if 'pr' in cfg.eval_metrics:
            print('TODO')
        # else:
        #     raise NotImplementedError

        
    # weight_dtype = torch.float32
    # if config.mixed_precision == 'fp16':
    #     weight_dtype = torch.float16
    # elif config.mixed_precision == 'bf16':
    #     weight_dtype = torch.bfloat16

    # fmgan.to(accelerator.device, dtype=weight_dtype)
    
    # -----------------------------------------------------------------------------
    # Dataset, DataLoader
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

    eval_real_ds = ImageDataset(root=cfg.dataset.root,
                                sub_dir=cfg.dataset.eval_real.sub_dir,
                                img_range=cfg.dataset.eval_fake.range,)

    recon1_dl = DataLoader(recon1_ds, batch_size=cfg.training.batch_size, 
                          shuffle=True, pin_memory=True, num_workers=cfg.training.loader_workers,
                          prefetch_factor=cfg.training.prefetch_factor)

    recon2_dl = DataLoader(recon2_ds, batch_size=cfg.training.batch_size,
                          shuffle=True, pin_memory=True, num_workers=cfg.training.loader_workers,
                          prefetch_factor=cfg.training.prefetch_factor)

    disen_dl = DataLoader(disen_ds, batch_size=cfg.training.batch_size,
                          shuffle=True, pin_memory=True, num_workers=cfg.training.loader_workers,
                          prefetch_factor=cfg.training.prefetch_factor)

    eval_fake_dl = DataLoader(eval_fake_ds, batch_size=cfg.eval.batch_size, 
                        shuffle=False, pin_memory=True, num_workers=cfg.eval.loader_workers,
                          prefetch_factor=cfg.eval.prefetch_factor)

    eval_real_dl = DataLoader(eval_real_ds, batch_size=cfg.eval.batch_size, 
                        shuffle=False, pin_memory=True, num_workers=cfg.eval.loader_workers,
                          prefetch_factor=cfg.eval.prefetch_factor)


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
    # Preparation
    # -----------------------------------------------------------------------------
    generator, discriminator, extractor, g_optimizer, d_optimizer, recon1_dl, recon2_dl, disen_dl, eval_fake_dl, eval_real_dl = accelerator.prepare(generator, discriminator, extractor, g_optimizer, d_optimizer, recon1_dl, recon2_dl, disen_dl, eval_fake_dl, eval_real_dl)

    fmgan = FMGAN(cfg, generator, discriminator)

    recon1_dl = sample_data(recon1_dl)
    recon2_dl = sample_data(recon2_dl)
    disen_dl = sample_data(disen_dl)

    logger.info(f"{'Running training':*^20}")
    logger.info(f"  reconstruction 1 num examples = {len(recon1_ds)}")
    logger.info(f"  reconstruction 2 num examples = {len(recon2_ds):*^20}")
    logger.info(f"  disentanglement num examples = {len(disen_ds):*^20}")

    step = 0
    cfg.total_step = cfg.training.p1_total_step+cfg.training.p2_total_step

    if exists(ckpt.get('enc_t', None)):
        accelerator.unwrap_model(generator).enc_t.load_state_dict(ckpt['enc_t'], strict=False)
    if exists(ckpt.get('enc_w', None)):
        accelerator.unwrap_model(generator).enc_w.load_state_dict(ckpt['enc_w'], strict=False)
    if exists(ckpt.get('enc_wplus', None)):
        accelerator.unwrap_model(generator).enc_wplus.load_state_dict(ckpt['enc_wplus'], strict=False)
    if exists(ckpt.get('g', None)):
        accelerator.unwrap_model(generator).stylegan.load_state_dict(ckpt['g'], strict=False)
    if exists(ckpt.get('d', None)) and exists(accelerator.unwrap_model(discriminator)):
        accelerator.unwrap_model(discriminator).load_state_dict(ckpt['d'], strict=False)
    # optimizer는 따로 load할 필요가 없나?
    if exists(ckpt.get('g_optimizer', None)):
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    if exists(ckpt.get('d_optimizer', None)):
        d_optimizer.load_state_dict(ckpt['d_optimizer'])


    # -----------------------------------------------------------------------------
    # Prepare evaluation real dataset moments
    # -----------------------------------------------------------------------------
    if 'fid50k' in cfg.eval_metrics:
        num_generate = 50_000
    else:
        raise NotImplementedError
    
    mu, sigma = prepare_real_moments(accelerator, eval_real_dl, extractor, num_generate, True)
    fmgan.mu = mu
    fmgan.sigma = sigma


    # -----------------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------------
    progress_bar = tqdm(range(0, cfg.training.p1_total_step), 
						disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Phase 1')
    
    for idx in progress_bar:
        # 재시작하는 경우
        step = idx + cfg.training.start_step
        if step > cfg.training.p1_total_step:
            print('phase1 done')
            break
        
        fmgan.train()
        losses = train_one_step(fmgan, accelerator, recon1_dl, disen_dl, d_optimizer, g_optimizer)
        post_train_step(accelerator, cfg, losses, fmgan, extractor, g_optimizer, d_optimizer, step, eval_fake_dl, eval_real_dl)
        progress_bar.set_postfix(**losses)
    # -----------------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------------
    progress_bar = tqdm(range(cfg.training.p2_total_step), 
                    disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Phase 2')

  
    # -----------------------------------------------------------------------------
    # Update learning rate
    # -----------------------------------------------------------------------------
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = cfg.phase2_lr #0.001

    for param_group in d_optimizer.param_groups:
        param_group['lr'] = cfg.phase2_lr #0.001

    for idx in progress_bar:
        step += idx
        if step >= cfg.total_step:
            print('phase2 done')
            break
        
        fmgan.train()
        losses = train_one_step(fmgan, accelerator, recon2_dl, disen_dl, d_optimizer, g_optimizer)
        post_train_step(accelerator, cfg, losses, fmgan, step, eval_fake_dl, eval_real_dl)
        progress_bar.set_postfix(**losses)
    # main process에 있는 tracker를 중지하는 기능
    # 학습이 끝난 뒤 반드시 설정
    accelerator.end_training()


if __name__ == '__main__':
    main()