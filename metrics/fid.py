import numpy as np
import scipy.linalg
from torchvision.datasets import ImageFolder
# from . import metric_utils

#----------------------------------------------------------------------------

# def compute_fid(opts, max_real, num_gen, swav=False, sfid=False):
#     # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
#     detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
#     detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

#     mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
#         opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
#         rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, swav=swav, sfid=sfid).get_mean_cov()

#     mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
#         opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
#         rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, swav=swav, sfid=sfid).get_mean_cov()

#     if opts.rank != 0:
#         return float('nan')

#     m = np.square(mu_gen - mu_real).sum()
#     s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
#     fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
#     return float(fid)




import sys
sys.path.append('/workspace/gan/3D-FM-GAN/model')
sys.path.append('/workspace/gan/3D-FM-GAN')
import torch
import torch.nn
from model.networks import *
from model.dataset import *
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
from scipy import linalg
import math


"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""
def frechet_inception_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        "Training and test mean vectors have different lengths."
    assert sigma1.shape == sigma2.shape, \
        "Training and test covariances have different dimensions."

    
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)



def calc_moments_fake(fake_feats, num_generate):
    acts = fake_feats.detach().cpu().numpy()[:num_generate].astype(np.float64)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

def calc_moments_real(accelertor, data_loader, eval_model, num_generate, batch_size, quantize):
    # real data에서 얻은 데이터
    # 
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
    

def calculate_fid(accelerator,
                  cfg,
                  eval_model,
                  eval_real_dl,
                  fake_feats,
                  num_generate=50_000,
                  pre_cal_mean=None,
                  pre_cal_std=None,
                  quantize=True):
    
    eval_model.eval()

    effective_batch = cfg.training.batch_size * cfg.num_gpus

    # calc moments from real data 
    if pre_cal_mean is not None and pre_cal_std is not None:
        m1, s1 = pre_cal_mean, pre_cal_std
    else:
        m1, s1 = calc_moments_real(accelerator, data_loader=eval_real_dl,
                                   eval_model=eval_model,
                                   num_generate=num_generate,
                                   batch_size=effective_batch,
                                   quantize=quantize)

    # calc moments from fake data
    m2, s2 = calc_moments_fake(fake_feats, num_generate)

    fid_value = frechet_inception_distance(m1, s1, m2, s2)
    return fid_value, m1, s1
                        





# if __name__ == '__main__':
#     cfg = OmegaConf.create({"wplus_num_layers" : 50,
#                              "output_size" : 256,
#                              "ckpt_path" : '/workspace/gan/3D-FM-GAN/pretraiend/550000.pt',
#                              "truncation" : 1.0,
#                              "truncation_mean" : 4096})

#     device = 'cuda'
#     fmgenerator = FMGenerator(cfg).to(device)
#     fmgenerator.eval()

#     if cfg.truncation < 1:
#         with torch.inference_mode():
#             mean_latent = fmgenerator.stylegan.mean_latent(cfg.truncation_mean)
#     else:
#         mean_latent = None


#     inception = nn.DataParallel(load_patched_inception_v3()).to(device)
#     inception.eval()

#     features = extract_feature_from_samples(
#         fmgenerator.stylegan, inception, cfg.truncation, mean_latent, cfg.batch, args.n_sample, device
#     ).numpy()
#     print(f"extracted {features.shape[0]} features")

#     sample_mean = np.mean(features, 0)
#     sample_cov = np.cov(features, rowvar=False)

#     with 
    
    