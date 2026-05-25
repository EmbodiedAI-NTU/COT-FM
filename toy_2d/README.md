# Toy 2D Point Cloud (COT-FM)

Unconditional 2D point-cloud generation comparing FM / MeanFlow / OT-CFM / COT-FM on 5-Gaussians, Two Moons, and Checkerboard (paper §4.1, Table 1).

## Env

Uses [uv](https://docs.astral.sh/uv/) (Python >= 3.13):

```bash
uv sync
```

## Run

Everything lives in `main.py`. Pick an experiment by uncommenting its block in `main()`, then:

```bash
uv run python main.py
```

Each block calls `train_flow(X0, X1, model_type=...)` with `model_type` in `flow`, `meanflow`, `otcfm`, `cotfm`. Source/target are chosen per case from `GaussianMixture5`, `TwoMoons`, `CheckerboardGrid`; it prints Wasserstein distance and curvature and saves trajectory plots.

## Data & assets

None — the 2D datasets are generated in code; no downloads or pretrained models.
