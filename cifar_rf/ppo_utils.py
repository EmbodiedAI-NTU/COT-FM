import os
import torch
from torch.distributions import Normal
import logging
from typing import Dict


# ============================================================================
# Configuration
# ============================================================================

class NoiseConfig:
    # Number of clusters the initial-noise statistics are conditioned on.
    num_classes = 100
    batch_size = 1024
    device = 'cuda'

    num_reflow_samples = 600000
    reflow_data_path_0 = "./logs/1_rectified_flow/eval/ckpt_6_reflow_data_merged_0.pth"

    flow_model_path = "./logs/1_rectified_flow/checkpoints-meta/checkpoint_ot_no_cluster_longer.pth"
    # Per-class reverse statistics used by the learned initial distribution.
    gaussians_path = "./logs/1_rectified_flow/eval/reverse_stats_cov_100_5.pth"
    data_path = f"./cifar10_{num_classes}_cluster.pt"

    # Initial-noise mode: standard N(0, I) if `gaussian`, per-class diagonal
    # gaussians if `std`, otherwise per-class full-covariance gaussians.
    gaussian = False
    std = False


ppo_config = NoiseConfig()

# The per-class reverse statistics are only needed by the learned-init modes.
# Load them lazily so that `gaussian=True` (standard N(0, I) init) runs without them.
gaussians = None
dists = None


def _load_gaussians():
    """Load per-class reverse statistics and build the sampling distributions."""
    global gaussians, dists
    if dists is None:
        gaussians = torch.load(ppo_config.gaussians_path)
        if not ppo_config.std:
            dists = [torch.distributions.MultivariateNormal(gaussians["means"][i], gaussians["covs"][i])
                     for i in range(ppo_config.num_classes)]
    return gaussians, dists

def load_ppo_resources(device: str = 'cuda') -> Dict:
    """Load the clustered CIFAR-10 images and their cluster assignments."""
    data_path = ppo_config.data_path
    logging.info(f"Loading clustered dataset from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Clustered dataset not found at {data_path}. "
            f"Set --config.eval.data_path to point at it.")
    ppo_data = torch.load(data_path, map_location='cpu')

    resources = {
        'ppo_data': ppo_data,
        'class_to_indices': ppo_data['class_to_indices'],
        'device': device
    }
    logging.info("Clustered dataset loaded successfully")
    return resources


def sample_noise(class_labels: torch.Tensor, device: str):
    """Draw the initial noise for each sample from its cluster's distribution."""
    class_labels = class_labels.cpu()
    if ppo_config.gaussian:
        noise = torch.randn(class_labels.shape[0], 3*32*32).to(device)
        return noise
    gaussians, dists = _load_gaussians()
    if ppo_config.std:
        means, stds = gaussians["means"][class_labels], gaussians["stds"][class_labels]
        dist = Normal(means.to(device), stds.to(device))
        noise = dist.rsample()
    else:
        noise = []
        for class_label in class_labels:
            noise.append(dists[class_label.item()].rsample().to(device))
        noise = torch.stack(noise, dim=0)
    return noise




