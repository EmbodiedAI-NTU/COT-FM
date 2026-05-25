# COT-FM: Cluster-wise Optimal Transport Flow Matching

![COT-FM teaser](assets/teaser.png)

COT-FM is a general, plug-and-play framework that reshapes the probability path in Flow Matching (FM) for faster and more reliable generation. FM models often produce curved trajectories due to random or batch-wise couplings, which increase discretization error and reduce sample quality. COT-FM fixes this by clustering target samples and assigning each cluster a dedicated source distribution obtained by reversing pretrained FM models. This divide-and-conquer strategy yields more accurate local transport and significantly straighter vector fields, all without changing the model architecture. It consistently accelerates sampling and improves generation quality across 2D datasets, image generation benchmarks, and robotic manipulation tasks.

## Paper

**COT-FM: Cluster-wise Optimal Transport Flow Matching**
Chiensheng Chiang\*, Kuan-Hsun Tu\*†, Jia-Wei Liao\*, Cheng-Fu Chou, Tsung-Wei Ke
National Taiwan University (\* Equal contribution, † Corresponding author)

- arXiv: [2603.13395](https://arxiv.org/abs/2603.13395)
- Project page: [embodiedai-ntu.github.io/cotfm](https://embodiedai-ntu.github.io/cotfm)

## Experiments

| Experiment | Dataset | Folder | Train | Eval |
|---|---|---|---|---|
| Unconditional 2D point cloud (§4.1, Table 1) | 5-Gaussians / Two Moons / Checkerboard | [`toy_2d`](toy_2d) | [`main.py`](toy_2d/main.py) | [`main.py`](toy_2d/main.py) |
| Unconditional image gen, Rectified Flow (§4.2, Table 2) | CIFAR-10 | [`cifar_rf`](cifar_rf) | [`main.py`](cifar_rf/main.py) | [`main.py`](cifar_rf/main.py) |
| Unconditional image gen, OT-CFM (§4.2, Table 2) | CIFAR-10 | [`cifar_otcfm`](cifar_otcfm) | [`train_cifar10.py`](cifar_otcfm/train_cifar10.py) | [`compute_fid_multi_gpu.py`](cifar_otcfm/compute_fid_multi_gpu.py) |
| Unconditional image gen, MeanFlow (§4.2, Table 2) | CIFAR-10 | [`cifar_meanflow`](cifar_meanflow) | [`train.py`](cifar_meanflow/train.py) | [`train.py`](cifar_meanflow/train.py) |
| Conditional image gen (§4.3, Table 3) | ImageNet 256×256 (SiT-B/2, SiT-B/4) | [`imagenet`](imagenet) | [`train_cotfm.py`](imagenet/train_cotfm.py) | [`evaluate.py`](imagenet/evaluate.py) |

`cifar_rf` selects train/eval via `--mode {train,eval}`; `cifar_meanflow` evaluates with `--eval_only` (see [`scripts/`](cifar_meanflow/scripts)); `toy_2d` runs training, sampling, and metrics from a single `main.py`. Each folder has its own README with setup and run instructions.

## Citation

```bibtex
@article{chiang2026cotfm,
  title={COT-FM: Cluster-wise Optimal Transport Flow Matching},
  author={Chiang, Chiensheng and Tu, Kuan-Hsun and Liao, Jia-Wei and Chou, Cheng-Fu and Ke, Tsung-Wei},
  journal={arXiv preprint arXiv:2603.13395},
  year={2026}
}
```

## License

Released under [CC BY-NC 4.0](LICENSE) (non-commercial) — © 2026 Chiensheng Chiang, Kuan-Hsun Tu, Jia-Wei Liao, Cheng-Fu Chou, Tsung-Wei Ke (National Taiwan University).

Subfolders derived from upstream codebases keep their original `LICENSE` files; those upstream terms also apply to the corresponding code.
