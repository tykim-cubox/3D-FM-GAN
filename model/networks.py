import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import resnet18
from encoders.psp_encoders import GradualStyleEncoder
from stylegan2.model import Discriminator, Generator

from lpips import LPIPS
from einops import rearrange
from losses import IDLoss, ContentLoss, g_hinge, g_nonsaturating_loss, d_hinge, d_logistic_loss

gan_loss_list = {'hinge' : [g_hinge, d_hinge],
                'basic' : [g_nonsaturating_loss, d_logistic_loss]}

class EncoderT(nn.Module):
    """
    tensor
    """
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=False)
        # [1, 3, 256, 256] -> [1, 512, 8, 8]
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # 논문에 언급되는 shape 512x4x4 와 맞추기 위해서 
        # 아니면 pooling을 해야
        # reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=4, w1=4).shape
        self.final_layer = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        
    def forward(self, x):
        x = self.features(x)
        x = self.final_layer(x)
        return x

class EncoderW(nn.Module):
    """
    modulation
    """
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # [1, 3, 256, 256] -> [1, 512, 1, 1]
        x = self.features(x)
        # [B 512 1 1] > [B 512]
        return rearrange(x, 'b c h w -> b (h w) c')

class GradualStyleEncoder18(GradualStyleEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c1 = x
            elif i == 5:
                c2 = x
            elif i == 7:
                c3 = x

        for j in range(self.coarse_ind):
            # print(c3.shape)
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out

class EncoderWplus(nn.Module):
    def __init__(self, num_layers=50, mode='ir', output_size=256):
        super().__init__()
        # self.features = GradualStyleEncoder(num_layers, mode, output_size, input_nc=3)
        self.features = GradualStyleEncoder18(num_layers, mode, output_size, input_nc=3)

    def forward(self, x):
        x = self.features(x)
        return x

class FMGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Models
        self.enc_t = EncoderT()
        self.enc_w = EncoderW()
        self.enc_wplus = EncoderWplus(num_layers=args.wplus_num_layers,
                                      output_size=args.output_size)
        
        self.stylegan = Generator(args.output_size, 512, 8)

        # if args.ckpt_path is not None:
        #     self.load_weights(args.ckpt_path)

    def forward(self, orig_img, rendered_img):
        """
        Returns:
            images : [B, C, H, W]
        """
        t = self.enc_t(rendered_img)
        w = self.enc_w(rendered_img)
        wplus = self.enc_wplus(orig_img)

        # print(w.shape, wplus.shape) : torch.Size([3, 1, 512]) torch.Size([3, 14, 512])
        w_final = w * wplus
        # t : [B, 512, 4, 4]
        # w_final : [B, 14, 512]
        # randomize_noise를 넣어야 하나?
        images, result_latent = self.stylegan([w_final], tensor_input=t,
                                            input_is_latent=True,
                                            randomize_noise=False,
                                            return_latents=True,
                                            input_is_tensor=True)
        # images : [B, 3, 256, 256]
        # result_latent : [B, 18, 512]
        return images

    def cuda(self):
        self.enc_t.to('cuda')
        self.enc_w.to('cuda')
        self.enc_wplus.to('cuda')
        self.stylegan.to('cuda')
    
    def device(self):
        # 그냥 실행하면 일단 모두 cpu에 올라가 있음
        return {'stylegan': next(self.stylegan.parameters()).device, 'enc_t' : next(self.enc_t.parameters()).device, 'enc_w' : next(self.enc_w.parameters()).device, 'enc_wplus' : next(self.enc_wplus.parameters()).device,}


    # def load_weights(self, g_ckpt_path=None, enc_t_ckpt_path=None, enc_w_ckpt_path=None, enc_wplus_path=None):
    #     g_ckpt = torch.load(g_ckpt_path)
    #     self.stylegan.load_state_dict(g_ckpt['g_ema'], strict=False)


class FMGAN(nn.Module):
    def __init__(self, cfg, generator, discriminator, lpips_loss, id_loss, l1_loss, content_loss):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        # self.generator = FMGenerator(cfg.model)

        # # 학습의 경우 Discriminator 필요
        # if cfg.model.disc:
        #     self.discriminator = Discriminator(cfg.model.input_size)
            
            # if cfg.model.ckpt_path is not None:
            #     ckpt = torch.load(cfg.model.ckpt_path)
            #     self.discriminator.load_state_dict(ckpt['d'], strict=False)

        # Losses
        self.lpips_lambda = cfg.training.lpips_lambda
        self.l1_lambda = cfg.training.l1_lambda
        self.id_lambda = cfg.training.id_lambda
        self.content_lambda = cfg.training.content_lambda
        


        # device = self.generator.device()['stylegan']
        self.g_loss = gan_loss_list[cfg.training.loss_type][0]
        self.d_loss = gan_loss_list[cfg.training.loss_type][1]
        # LPIPS, IDLoss는 네트워크에 의존하기 때문에 eval() 모드로 
        # self.lpips_loss = LPIPS(net=cfg.training.lpips_type).eval().to(device)
        # self.id_loss = IDLoss(cfg.training).eval().to(device)
        # self.l1_loss = torch.nn.L1Loss().to(device)
        # self.content_loss = ContentLoss().to(device)
        self.lpips_loss = lpips_loss
        self.id_loss = id_loss
        self.l1_loss = l1_loss
        self.content_loss = content_loss


    def disentangle_loss(self, p1, r1, m1, p2, r2, m2, model):
        loss = 0.0
        if model == 'd':
            p1_hat = self.generator(p1, r2)
            self.disen_p1_hat = p1_hat
            p1_target = p2
            
            p1_fake_logit = self.discriminator(p1_hat.detach())
            p1_real_logit = self.discriminator(p1_target)
            loss += self.d_loss(p1_fake_logit, p1_real_logit)

            p2_hat = self.generator(p2, r1)
            self.disen_p2_hat = p2_hat
            p2_target = p1
            
            p2_fake_logit = self.discriminator(p2_hat.detach())
            p2_real_logit = self.discriminator(p2_target)
            loss += self.d_loss(p2_fake_logit, p2_real_logit)

        if model == 'g':
            # p1_hat = self.generator(p1, r2)
            p1_hat = self.disen_p1_hat
            p1_target = p2

            p1_fake_logit = self.discriminator(p1_hat)
            loss += self.g_loss(p1_fake_logit)
            loss += self.id_lambda * self.id_loss(p1_target , p1_hat).mean()
            loss += self.l1_lambda * self.l1_loss(p1_target , p1_hat)
            loss += self.lpips_lambda * self.lpips_loss(p1_target , p1_hat).mean()
            loss += self.content_lambda * self.content_loss(p1_hat, r2, m2).mean() # m2와 r2를 이용

            # p2_hat = self.generator(p2, r1)
            p2_hat = self.disen_p2_hat
            p2_target = p1

            p2_fake_logit = self.discriminator(p2_hat)
            loss += self.g_loss(p2_fake_logit)
            loss += self.id_lambda * self.id_loss(p2_target , p2_hat).mean()
            loss += self.l1_lambda * self.l1_loss(p2_target , p2_hat)
            loss += self.lpips_lambda * self.lpips_loss(p2_target , p2_hat).mean()
            loss += self.content_lambda * self.content_loss(p2_hat, r1, m1).mean()

        return loss
        
    def reconstruction_loss(self, p1, r1, model):
        # loss = 0.0
        if model == 'd':
            p1_hat = self.generator(p1, r1)
            self.recon_p1_hat = p1_hat
            print(self.recon_p1_hat.requires_grad)
            fake_logit = self.discriminator(p1_hat.detach())
            real_logit = self.discriminator(p1)
            loss = self.d_loss(fake_logit, real_logit)

        if model == 'g':
            # 여기서 OOM은 배치를 16으로 줄여서 해결
            p1_hat = self.generator(p1, r1)
            # p1_hat = self.recon_p1_hat
            print('test', p1_hat.requires_grad)
            fake_logit = self.discriminator(p1_hat)
            # fake_logit : [16, 1]
            loss =  self.g_loss(fake_logit)
            # self.id_loss(p1, p1_hat) [16] -> []
            idloss =  self.id_lambda * self.id_loss(p1, p1_hat).mean()
            loss += idloss
            # []
            l1loss = self.l1_lambda * self.l1_loss(p1, p1_hat)
            loss += l1loss
            # OOM
            lpipsloss = self.lpips_lambda * self.lpips_loss(p1, p1_hat).mean()
            loss += lpipsloss
        return loss
        
        
    # @torch.inference_mode()
    # def inference(self, pic_img, coeff_edit_args):
    #     # Face Recon
    #     pic_img = pic_img.to(self.fr_model.device)
    #     origin_coeff = self.fr_model.net_recon(pic_img)

    #     new_coeff = self.edit_coeff(origin_coeff, coeff_edit_args)
    @torch.inference_mode()
    def inference(self, pic_img, renderend_img):
        return self.generator(pic_img, renderend_img)
# opt

# from Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper
# class Renderer(nn.Module):
#     def __init_(self, render_opt):
#         self.fr_model = ReconNetWrapper(net_recon=render_opt.net_recon,
#                                         use_last_fc=render_opt.use_last_fc,
#                                         input_path=render_opt.init_path)
#         self.


#     def forward(self, input):
#         ...
#     def edit_coeff(self, origin_coeff, coeff_edit_args):
#         batch_size = origin_coeff.shape(0)
#         if coeff_edit_args == 'random_exp':
#             new_coeff = torch.randn(batch_size, 64)
#             origin_coeff[:, 80:80+64] = new_coeff
#         elif coeff_edit_args == 'random_pose':
#             new_coeff = torch.randn(batch_size, 3)
#             origin_coeff[:, 80+64+80:80+64+80+3] = new_coeff

#         elif coeff_edit_args == 'random_illumnination':
#             new_coeff = torch.randn(batch_size, 27)
#             origin_coeff[:, 80+64+80+3:80+64+80+3+27] = new_coeff

#         elif coeff_edit_args == 'random_all':
#             new_coeff = torch.randn(batch_size, 27)
#         else:
#             ...

#         return new_coeff

    


  