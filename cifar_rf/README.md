# CIFAR-10, Rectified Flow (COT-FM)

Unconditional CIFAR-10 image generation on the Rectified Flow backbone (paper §4.2, Table 2). Built on the RectifiedFlow / score_sde codebase.

## Env

```bash
conda env create -f environment.yml   # creates env "rectflow"
conda activate rectflow
```

## Run

Entry point is `main.py` (absl flags `--config`, `--workdir`, `--mode {train,eval,reflow}`); configs live in `configs/rectified_flow/`.

```bash
# cluster CIFAR-10 (DINO + k-means) into cluster-wise source distributions
python clustering.py

# train base 1-Rectified-Flow
python main.py --config configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --workdir logs/cifar_rf --mode train

# reflow / COT-FM fine-tune
python main.py --config configs/rectified_flow/cifar10_rf_gaussian_reflow_train.py --workdir logs/cifar_rf_reflow --mode reflow
```

### Evaluation

`--mode eval` dispatches on `--config.eval.eval_mode`:

| `eval_mode` | what it does |
| --- | --- |
| `train_fid` | FID + KID + IS against the 50k CIFAR-10 **train** split |
| `test_fid` | FID against the 10k CIFAR-10 **test** split |
| `generate` | sample grids, one row per cluster; no metrics |
| `reflow` | generate and dump (noise, image) pairs into `<workdir>/<eval_folder>/` for the reflow stage |
| `reverse` | run the reverse ODE to estimate the per-cluster noise statistics (output path is hardcoded in `find_reverse`) |

Metrics come from `torchmetrics`: `FrechetInceptionDistance(feature=2048, normalize=True)`,
`KernelInceptionDistance(subset_size=1000, feature=2048, normalize=True)` and
`InceptionScore(normalize=True)`. A running FID estimate is logged after every batch;
only the final line is the reported number.

The initial noise is selected by `--config.eval.gaussian` / `--config.eval.std`:

```bash
# COT-FM: per-cluster full-covariance gaussians (the default, gaussian=False std=False).
# Cluster labels are drawn with probability proportional to each cluster's size.
python main.py \
  --config configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
  --workdir /path/to/workdir \
  --mode eval --eval_folder eval_cotfm \
  --config.eval.eval_mode=train_fid \
  --config.eval.enable_sampling=True \
  --config.eval.begin_ckpt=8 --config.eval.end_ckpt=8 \
  --config.eval.batch_size=500 \
  --config.eval.flow_model_path=/path/to/checkpoint.pth \
  --config.eval.gaussians_path=/path/to/reverse_stats.pth \
  --config.eval.data_path=/path/to/cifar10_cluster.pt \
  --config.sampling.use_ode_sampler=euler --config.sampling.sample_N=50

# Baseline: standard N(0, I) initial noise. `gaussians_path` is then unused.
python main.py \
  --config configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
  --workdir /path/to/workdir \
  --mode eval --eval_folder eval_gaussian \
  --config.eval.eval_mode=train_fid \
  --config.eval.gaussian=True \
  --config.eval.enable_sampling=True \
  --config.eval.begin_ckpt=8 --config.eval.end_ckpt=8 \
  --config.eval.batch_size=500 \
  --config.eval.flow_model_path=/path/to/checkpoint.pth \
  --config.eval.data_path=/path/to/cifar10_cluster.pt \
  --config.sampling.use_ode_sampler=euler --config.sampling.sample_N=50
```

Notes:

- `--config.sampling.use_ode_sampler` is `euler` (fixed `sample_N` steps) or `rk45`
  (adaptive, `ode_tol`). With `rk45` the batch size **must divide the dataset size**,
  because the ODE solver reshapes to a fixed sampling shape.
- `--config.eval.batch_size` is also the metric batch size. 500 fits in 24 GB with
  `euler`; the 1024 default needs a larger card.
- Nothing in the eval path seeds the RNG, so FID varies slightly between runs.
- `--config.eval.{flow_model_path,gaussians_path,data_path}` default to `'no'`, which
  means "keep the built-in default in `ppo_utils.NoiseConfig`".

## Data & assets

**Produced by the pipeline:**
- `cifar10_100_cluster.pt` — CIFAR-10 images + DINO features + cluster assignments (from `clustering.py`; pass it as `--config.eval.data_path`).
- per-cluster Gaussian stats from the reverse ODE (`--mode eval --config.eval.eval_mode=reverse`; pass as `--config.eval.gaussians_path`) and base Rectified Flow checkpoints (`--mode train`).
- `/path/to/eval_ppo_training` (in `run_lib.py`) — just an output directory; set any writable path.

**Download manually:**
- `assets/stats/cifar10_stats.npz` — TF-Inception FID reference stats, from [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch). Only used by the `run_lib_pytorch.py` evaluation path (`tfgan`), which `main.py` selects when `config.data.dataset` contains `pytorch`; the configs in `configs/rectified_flow/` use the `torchmetrics` path above instead.
- `cotfm_rf.pth` — COT-FM fine-tuned checkpoint, from [Google Drive](https://drive.google.com/drive/folders/1Jy7bhbI6LtwehKNY8tCx3OY1HNS6xcuO?usp=sharing).
