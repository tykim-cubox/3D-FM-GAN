import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.insight_face.model_irse import Backbone

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p_hat, r_tgt, m):
        return self.mse(m*p_hat, m*r_tgt)
        
class IDLoss(nn.Module):
    def __init__(self, cfg):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        try:
            self.facenet.load_state_dict(torch.load(cfg.id_model_path))
        except IOError:
            self.facenet.load_state_dict(torch.load('/apdcephfs/share_916081/amosyhliu/pretrained_models/model_ir_se50.pth'))

        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        return losses / n_samples
    

# logistic and non-saturating loss
def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

# Hinge Losses
def g_hinge(d_logit_fake):
    return -torch.mean(d_logit_fake)

def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))