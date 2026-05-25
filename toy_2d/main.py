import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import ot
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from tqdm import trange
from torch.func import jvp
from torch.utils.data import DataLoader, TensorDataset
import random


device = "cuda" if torch.cuda.is_available() else "cpu"


def cal_discrete_curvature(X, eps=1e-12):
    """
    X: np.ndarray, shape (B, T, D)
    回傳: (B,) 每筆資料的平均離散曲率
    """
    assert X.ndim == 3, f"Expected (B, T, D), got {X.shape}"

    # 一階差分 along time
    dr = np.diff(X, axis=1)                     # (B, T-1, D)
    s_len = np.linalg.norm(dr, axis=2)          # (B, T-1)

    # 避免 0 長度造成除 0
    s_len_safe = np.maximum(s_len, eps)         # (B, T-1)
    dr_norm = dr / s_len_safe[:, :, None]       # (B, T-1, D)

    # 二階差分（方向變化）
    ddr = np.diff(dr_norm, axis=1)              # (B, T-2, D)

    # 這裡 s_len 是 2D，要用 2D 切法
    s_prev = s_len[:, :-1]                      # (B, T-2)
    s_next = s_len[:,  1:]                      # (B, T-2)
    ds_mid = (s_prev + s_next) / 2              # (B, T-2)
    ds_mid_safe = np.maximum(ds_mid, eps)

    # 曲率 κ ≈ ||Δtangent|| / Δs
    kappa = np.linalg.norm(ddr / ds_mid_safe[:, :, None], axis=2)  # (B, T-2)
    return kappa.mean(axis=1).mean(axis=0)                   # (B,)



def cal_wasserstein_distance(X, Y):
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    m, n = X.shape[0], Y.shape[0]

    # cost matrix
    C = torch.cdist(X_t, Y_t, p=2) ** 2

    # uniform weights
    w0 = torch.full((m,), 1.0 / m)
    w1 = torch.full((n,), 1.0 / n)

    C_cpu = C.cpu().numpy()
    w0_cpu = w0.cpu().numpy()
    w1_cpu = w1.cpu().numpy()
    P = ot.emd(w0_cpu, w1_cpu, C_cpu)
    W2sq = (P * C_cpu).sum()
    return W2sq


# ========== Dataset definitions ==========
class Gaussian:
    def __init__(self, mean=[0, 0], sigma=0.5):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.sigma = sigma

    def sample(self, n):
        eps = torch.randn(n, 2) * self.sigma
        return self.mean + eps
    

class GaussianMixture2:
    def __init__(self, sigma=0.35):
        self.means = [
            torch.tensor([5, 5], dtype=torch.float32),
            torch.tensor([10, -5], dtype=torch.float32),
        ]
        self.sigma = sigma

    def sample(self, n):
        idx = torch.randint(0, len(self.means), (n,))
        eps = torch.randn(n, 2) * self.sigma
        mu = torch.stack(self.means)[idx]
        return mu + eps


class GaussianMixture5:
    def __init__(self, spread=4.0, sigma=0.35):
        self.means = [
            torch.tensor([10, spread * (i-3)], dtype=torch.float32)
            for i in range(1, 6)
        ]
        self.sigma = sigma

    def sample(self, n):
        idx = torch.randint(0, len(self.means), (n,))
        eps = torch.randn(n, 2) * self.sigma
        mu = torch.stack(self.means)[idx]
        return mu + eps


class CheckerboardGrid:
    """
    Uniform sampling on a checkerboard (black/white) grid.
    White cells have points; black cells are empty (or vice versa).
    All cells have equal size and are arranged symmetrically.
    """

    def __init__(
        self,
        grid_size=3,
        cell_size=2.0,
        include_white=True,
        center=(0.0, 0.0),
    ):
        """
        grid_size:      K → K×K checkerboard
        cell_size:      length of each square's side
        include_white:  True → white cells have points
                        False → black cells have points
        center:         (cx, cy) → center position of entire grid
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.include_white = include_white
        self.center = torch.tensor(center, dtype=torch.float32)

        # grid coordinates centered at origin before shift
        coords = torch.arange(grid_size) - (grid_size - 1) / 2
        centers = []
        valid_mask = []

        for i in coords:
            for j in coords:
                # checkerboard pattern: (i+j) even/odd
                is_white = ((i + j).abs() % 2 == 0)

                if include_white:
                    valid = is_white
                else:
                    valid = not is_white

                valid_mask.append(valid)

                centers.append(torch.tensor([
                    i * cell_size,
                    j * cell_size
                ], dtype=torch.float32))

        centers = torch.stack(centers)  # (K*K,2)
        self.centers = centers[torch.tensor(valid_mask)]
        self.num_cells = len(self.centers)

    def sample(self, n):
        if self.num_cells == 0:
            raise ValueError("Checkerboard has no active cells.")

        # pick random valid cells
        idx = torch.randint(0, self.num_cells, (n,))
        centers = self.centers[idx]

        # uniform noise inside the cell
        noise = (torch.rand(n, 2) - 0.5) * self.cell_size

        # apply center shift
        return centers + noise + self.center


class TwoMoons:
    def __init__(self, noise=0.06, rotate=1.3, shift=(10.0, 0.0)):
        self.noise = noise
        self.rotate = rotate
        self.shift = torch.tensor(shift, dtype=torch.float32)

    def sample_target(self, n):
        X, y = make_moons(n_samples=n, noise=self.noise)
        X = torch.tensor(X, dtype=torch.float32)
        rot = torch.tensor([
            [np.cos(self.rotate), -np.sin(self.rotate)],
            [np.sin(self.rotate),  np.cos(self.rotate)],
        ], dtype=torch.float32)
        X = X @ rot.T
        X = X + self.shift
        return X, torch.tensor(y)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ========== Model definitions ==========
class MLP(nn.Module):
    def __init__(self, hidden=96, out_dim=2):
        super().__init__()
        # 2 (x) + 1 (t) = 3
        in_dim = 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Swish(),
            nn.Linear(hidden, hidden),
            Swish(),
            nn.Linear(hidden, hidden),
            Swish(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


class MeanFlow(nn.Module):
    def __init__(self, in_dim=4, hidden=96, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, xt, r, t):
        return self.net(torch.cat([xt, r, t], dim=-1))



# ========== Flow Matching Trainer ==========
class FMTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.state = {"loss": []}

    def fit(self, dataloader0, dataloader1, n_epoch=50, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in trange(n_epoch):
            for (x0,), (x1,) in zip(dataloader0, dataloader1):
                x0, x1 = x0.to(self.device), x1.to(self.device)
                t = torch.rand(x0.shape[0], 1, device=self.device)
                xt = t * x1 + (1 - t) * x0
                ut = x1 - x0
                pred = self.model(xt, t)
                loss = self.loss_fn(pred, ut)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.state["loss"].append(loss.item())
        print(f"[Flow Matching] Final loss: {self.state['loss'][-1]:.4f}")


class MFTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()
        self.state = {"loss": []}

    def fit(self, dataloader0, dataloader1, n_epoch=50, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(n_epoch):
            for (x0,), (x1,) in zip(dataloader0, dataloader1):
                x0, x1 = x0.to(self.device), x1.to(self.device)
                batch_size = x0.shape[0]

                # --- sample t, r --- #
                t = torch.rand(batch_size, 1, device=self.device)
                r = torch.rand(batch_size, 1, device=self.device)
                # 確保 t > r
                t, r = torch.maximum(t, r), torch.minimum(t, r)

                z_t = (1 - t) * x1 + t * x0 
                v = x0 - x1 

                u_pred, dudt = jvp(
                    lambda xt, rr, tt: self.model(xt, rr, tt),
                    (z_t, r, t),
                    (v, torch.zeros_like(r), torch.ones_like(t))
                )

                u_tgt = v - (t - r) * dudt
                loss = ((u_pred - u_tgt.detach()) ** 2).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

            self.state["loss"].append(loss.item())

        print(f"[MeanFlow] Final loss: {self.state['loss'][-1]:.4f}")


class MBOTFMTrainer:
    """Mini-batch Optimal Transport Conditional Flow Matching (OT-CFM)"""
    def __init__(self, model, device, sigma=0.05):
        self.model = model
        self.device = device
        self.sigma = sigma
        self.loss_fn = nn.MSELoss()
        self.state = {"loss": []}

    def fit(self, dataloader0, dataloader1, n_epoch=100, lr=1e-3):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        for _ in trange(n_epoch):
            for (x0,), (x1,) in zip(dataloader0, dataloader1):
                x0, x1 = x0.to(self.device), x1.to(self.device)
                bs = x0.shape[0]

                # 1. Compute OT plan pi between (x0, x1)
                M = torch.cdist(x1, x0, p=2) ** 2  # cost matrix (squared Euclidean)
                M_np = M.detach().cpu().numpy()
                a, b = ot.unif(bs), ot.unif(bs)
                pi = ot.emd(a, b, M_np)

                # 2. Sample pairs (x0p, x1p) according to pi
                pi_flat = pi.flatten()
                pi_flat /= pi_flat.sum()
                idx = np.random.choice(bs * bs, bs, p=pi_flat)
                i, j = np.divmod(idx, bs)
                x0p, x1p = x0[j], x1[i]

                # 3. Sample t from Uniform [0,1]
                t = torch.rand(bs, 1, device=self.device)

                # 4. Compute meand interpolation
                mu_t = t * x1p + (1 - t) * x0p

                # 5. Add Gaussian noise
                x = mu_t + self.sigma * torch.randn_like(mu_t)

                # 6. Compute target velocity (x1 - x0)
                u_target = x1p - x0p

                # 7. Network predict
                u_pred = self.model(x, t)

                # 8. Compute Loss
                loss = self.loss_fn(u_pred, u_target)

                opt.zero_grad()
                loss.backward()
                opt.step()

            self.state["loss"].append(loss.item())

        print(f"[Mini-batch OT-CFM] Final loss: {self.state['loss'][-1]:.4f}")


class GlobalOTFMTrainer:
    """
    Global Optimal Transport Conditional Flow Matching (Global OT-CFM)
    """
    def __init__(self, model, device, sigma=0.05, recompute_each_epoch=False):
        self.model = model
        self.device = device
        self.sigma = sigma
        self.recompute_each_epoch = recompute_each_epoch
        self.loss_fn = nn.MSELoss()
        self.state = {"loss": []}
        self.pi_global = None  # 之後存全域 OT plan

    def _compute_global_ot(self, X0, X1):
        """
        X0, X1 : Tensor, shape (N, d)
        回傳 numpy array 的 π_global, shape (N, N)
        """
        X0 = X0.to(self.device)
        X1 = X1.to(self.device)
        N = X0.shape[0]
        # cost matrix: c_ij = ||x1_i - x0_j||^2
        M = torch.cdist(X1, X0, p=2) ** 2
        M_np = M.detach().cpu().numpy()
        a, b = ot.unif(N), ot.unif(N)
        pi = ot.emd(a, b, M_np)  # (N, N)
        return pi

    def fit(self, X0, X1, n_epoch=100, lr=1e-3, batch_size=512):
        """
        Parameters
        ----------
        X0 : Tensor (N, d)
            source dataset
        X1 : Tensor (N, d)
            target dataset
        """
        X0 = X0.to(self.device)
        X1 = X1.to(self.device)
        N = X0.shape[0]
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.pi_global = self._compute_global_ot(X0, X1)
        for _ in trange(n_epoch, desc="[Global OT-CFM]"):
            # 把全域 plan 攤平成機率分佈
            pi_flat = self.pi_global.flatten()
            pi_flat = pi_flat / pi_flat.sum()  # shape (N*N,)

            # 一個 epoch 我們就一直抽 batch 來 train
            # 抽多少看資料大小，這裡就照 N // batch_size 跑一圈
            n_iter = int(np.ceil(N / batch_size))
            epoch_losses = []

            for _ in range(n_iter):
                # 從全域 π 裡抽一批配對 index
                idx = np.random.choice(N * N, size=batch_size, replace=True, p=pi_flat)
                i, j = np.divmod(idx, N)   # i: target index, j: source index

                x0p = X0[j]   # (B, d)
                x1p = X1[i]   # (B, d)

                # 1. sample t ~ U[0,1]
                t = torch.rand(x0p.shape[0], 1, device=self.device)

                # 2. conditional interpolation
                #mu_t = (1 - t) * x0p + t * x1p

                # 3. add Gaussian noise
                # x = mu_t + self.sigma * torch.randn_like(mu_t)
                x = (1 - t) * x0p + t * x1p

                # 4. target velocity
                u_target = x1p - x0p

                # 5. predict
                u_pred = self.model(x, t)

                # 6. loss & update
                loss = self.loss_fn(u_pred, u_target)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_losses.append(loss.item())

            if epoch_losses:
                self.state["loss"].append(float(np.mean(epoch_losses)))

        print(f"[Global OT-CFM] Final loss: {self.state['loss'][-1]:.4f}")


# ===========================================
# Clustered Optimal Transport Flow Matching Trainer
# ===========================================
class COTFMTrainer:
    """Clustered Optimal Transport Flow Matching (COT-FM) Trainer"""
    def __init__(self, model, device, n_cluster=5, sigma=0.05):
        """
        Parameters
        ----------
        model : nn.Module
            neural network model u_theta(x, t)
        device : str
            device string ("cuda", "mps", or "cpu")
        n_cluster : int
            number of clusters for COT
        sigma : float
            Gaussian sampling std for cluster noise
        """
        self.model = model
        self.device = device
        self.n_cluster = n_cluster
        self.sigma = sigma
        self.loss_fn = nn.MSELoss()
        self.state = {"loss": []}

    def compute_clusters(self, X, Y):
        """
        Perform Clustered Optimal Transport on dataset X
        Returns a list of (mu_k, Sigma_k, pi_k, eps, Xk)
        """
        Y = Y.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_cluster, n_init="auto").fit(Y)
        labels = kmeans.labels_

        mean_vars = []
        for k in range(self.n_cluster):
            Xk = X[labels == k]
            Yk = Y[labels == k]
            nk = Xk.shape[0]
            mu_k = Xk.mean(0, keepdim=True)
            diff = Xk - mu_k
            Sigma_k = (diff.T @ diff) / nk
            mean_vars.append((mu_k, Sigma_k, Yk))
        self.mean_vars = mean_vars
        return mean_vars

    def prepare_COT_dataloader(self, mean_vars):
        """
        Perform Clustered Optimal Transport on dataset X
        Returns a list of (mu_k, Sigma_k, pi_k, eps, Xk)
        """
        X0 = []
        X1 = []
        for mu_k, Sigma_k, X1k in mean_vars:
            eps = mu_k + torch.randn_like(torch.Tensor(X1k)) @ torch.linalg.cholesky(Sigma_k + 1e-6 * torch.eye(X1k.shape[1], device=X1k.device)).T
            nk = X1k.shape[0]
            X1k = torch.Tensor(X1k)

            M = torch.cdist(eps, X1k) ** 2

            a, b = ot.unif(nk), ot.unif(nk)
            pi_k = ot.emd(a, b, M.detach().cpu().numpy())

            pi_flat = pi_k.flatten()
            pi_flat /= pi_flat.sum() + 1e-12
            idx = np.random.choice(nk * nk, nk, p=pi_flat)
            i, j = np.divmod(idx, nk)
            X0.append(eps[i])
            X1.append(X1k[j])

        X0 = torch.Tensor(np.concatenate(X0))
        X1 = torch.Tensor(np.concatenate(X1))
        return DataLoader(TensorDataset(X0, X1), batch_size=512, shuffle=True)

    # ===== Algorithm 2: COT-FM Training =====
    def fit(self, X1, n_epoch=1000, lr=1e-4):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        pipe = FMPipeline(model=self.model, device=self.device, n_step=1000)

        for k in trange(n_epoch, desc="[COT-FM] Training"):
            if k % 1000 == 0:
                inv_X1 = pipe.inversion(X1, 1000)["sample"]
                mean_vars = self.compute_clusters(inv_X1, X1)

            if k % 1 == 0:
                dataloader = self.prepare_COT_dataloader(mean_vars)

            noise = []
            for (x0_batch, x1_batch) in dataloader:
                x0_batch = x0_batch.to(self.device)
                x1_batch = x1_batch.to(self.device)

                # --- Sample t ~ U[0,1] --- #
                t = torch.rand(x1_batch.shape[0], 1, device=self.device)

                # --- COT-FM loss --- #
                xt = (1-t) * x0_batch + t * x1_batch
                u_pred = self.model(xt, t)
                loss = ((u_pred - (x1_batch - x0_batch)) ** 2).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                self.state["loss"].append(loss.item())
                noise.append(x0_batch.cpu())
        noise = torch.cat(noise, dim=0).numpy()
        inv_X1 = pipe.inversion(X1, 1000)["sample"]
        mean_vars = self.compute_clusters(inv_X1, X1)

        print(f"[COT-FM] Final loss: {self.state['loss'][-1]:.4f}")
        return noise


def visualize_ot_matching(X0, X1, pi, title="Optimal Transport Matching", n_lines=500):
    """
    Visualize 1-to-1 OT matching between X0 and X1 using the plan pi.
    
    Parameters
    ----------
    X0 : Tensor, shape (N, d)
        Source data.
    X1 : Tensor, shape (N, d)
        Target data.
    pi : np.ndarray, shape (N, N)
        Optimal transport plan matrix.
    title : str
        Title to save the plot with.
    n_lines : int
        Number of match lines to draw (for visibility).
    """
    if isinstance(X0, torch.Tensor):
        X0 = X0.detach().cpu().numpy()
    if isinstance(X1, torch.Tensor):
        X1 = X1.detach().cpu().numpy()

    N = pi.shape[0]
    # Flatten transport plan into probabilities
    pi_flat = pi.flatten()
    pi_flat = pi_flat / pi_flat.sum()
    
    # Sample matching pairs for visualization
    idx = np.random.choice(N * N, size=n_lines, p=pi_flat)
    i, j = np.divmod(idx, N)   # i: target index, j: source index
    
    plt.figure(figsize=(6, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=5, c="tab:blue", alpha=0.3, label="Source")
    plt.scatter(X1[:, 0], X1[:, 1], s=5, c="red", alpha=0.3, label="Target")

    lavender = "#B4AAFF"
    # Draw lines for each matching pair
    for t_idx, s_idx in zip(i, j):
        plt.plot(
            [X0[s_idx, 0], X1[t_idx, 0]],
            [X0[s_idx, 1], X1[t_idx, 1]],
            color=lavender, linewidth=0.3, alpha=0.2
        )

    plt.axis("equal")
    plt.xticks([]); plt.yticks([])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.box(False)
    # plt.legend()
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, transparent=True)
    print(f"Saved matching visualization: {title.replace(' ', '_')}.png")


def visualize_random_matching(X0, X1, title="Random Matching", n_lines=500):
    """
    Visualize random matching between X0 and X1.
    
    Parameters
    ----------
    X0 : Tensor or ndarray, shape (N, d)
        Source data.
    X1 : Tensor or ndarray, shape (N, d)
        Target data.
    title : str
        Title to save the plot with.
    n_lines : int
        Number of random match lines to draw.
    """
    if isinstance(X0, torch.Tensor):
        X0 = X0.detach().cpu().numpy()
    if isinstance(X1, torch.Tensor):
        X1 = X1.detach().cpu().numpy()

    N = X0.shape[0]
    assert X1.shape[0] == N, "The source and target datasets must have the same number of points."

    # Randomly sample matching indices
    s_idx = np.random.choice(N, size=n_lines, replace=True)  # source indices
    t_idx = np.random.choice(N, size=n_lines, replace=True)  # target indices

    plt.figure(figsize=(6, 6))
    plt.scatter(X0[:, 0], X0[:, 1], s=5, c="tab:blue", alpha=0.3, label="Source")
    plt.scatter(X1[:, 0], X1[:, 1], s=5, c="red", alpha=0.3, label="Target")

    lavender = "#B4AAFF"
    # Draw lines for each matching pair
    for si, ti in zip(s_idx, t_idx):
        plt.plot(
            [X0[si, 0], X1[ti, 0]],
            [X0[si, 1], X1[ti, 1]],
            color=lavender, linewidth=0.3, alpha=0.2
        )

    plt.axis("equal")
    plt.xticks([]); plt.yticks([])

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.box(False)
    plt.tight_layout()
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, transparent=True)
    print(f"Saved random matching visualization: {filename}")




# ========== Flow Matching Pipeline ==========
class FMPipeline:
    def __init__(self, model, device, n_step=20):
        self.model = model
        self.device = device
        self.n_step = n_step

    def __call__(self, x0):
        """Forward flow: from source → target"""
        x = x0.clone().to(self.device)
        traj = [x.detach().cpu().numpy()]

        for i in range(self.n_step):
            t = torch.full((x.shape[0], 1), i / self.n_step, device=self.device)
            u = self.model(x, t)
            x = x + u / self.n_step
            traj.append(x.detach().cpu().numpy())

        return {"sample": x.detach().cpu(), "traj": traj}

    def inversion(self, x1, n_step=1000):
        """Inverse flow: from target → source"""
        x = x1.clone().to(self.device)
        traj = [x.detach().cpu().numpy()]

        for i in range(n_step):
            t = torch.full((x.shape[0], 1), 1 - i / n_step, device=self.device)
            u = self.model(x, t)
            x = x - u / n_step
            traj.append(x.detach().cpu().numpy())

        return {"sample": x.detach().cpu(), "traj": traj}


class MFPipeline:
    def __init__(self, model, device, n_step=20):
        self.model = model
        self.device = device
        self.n_step = n_step

    def __call__(self, x0):
        x = x0.clone().to(self.device)
        traj = [x.detach().cpu().numpy()]

        dt = 1.0 / self.n_step
        for i in range(self.n_step):
            r = torch.full((x.shape[0], 1), i / self.n_step, device=x.device)
            t = r + dt
            u = self.model(x, r, t)
            x = x - u * dt
            traj.append(x.detach().cpu().numpy())

        return {"sample": x.detach().cpu(), "traj": traj}


# ===========================================
# Clustered Optimal Transport Flow Matching Sampling
# ===========================================
class COTFMPipeline:
    """COT-FM: Sampling according to Algorithm 3"""

    def __init__(self, model, device, clusters, n_step=20):
        """
        Parameters
        ----------
        model : nn.Module
            trained u_theta model
        device : str
            device string
        clusters : list
            list of tuples (mu_k, Sigma_k) obtained from COT
        n_step : int
            number of Euler integration steps
        """
        self.model = model
        self.device = device
        self.clusters = clusters  # [(mu_k, Sigma_k), ...]
        self.n_step = n_step

    def sample(self, noise, n_per_cluster=100):
        """
        Perform sampling from the learned COT-FM model.
        Returns a dict with sampled points and trajectories.
        """
        traj_all, eps_all, samples_all = [], [], []

        for mu_k, Sigma_k, _ in self.clusters:
            mu_k = mu_k.to(self.device)
            Sigma_k = Sigma_k.to(self.device)
            # eps = torch.Tensor(noise).to(self.device)
            # (Algorithm 3, line 2)
            eps = torch.randn(n_per_cluster, mu_k.shape[1], device=self.device)
            L = torch.linalg.cholesky(Sigma_k + 1e-6 * torch.eye(mu_k.shape[1], device=self.device))
            x = mu_k.to(self.device) + eps @ L.T
            x_init = x.clone()

            traj = [x.detach().cpu().numpy()]


            dt = 1.0 / self.n_step
            for i in range(self.n_step):
                r = torch.full((x.shape[0], 1), i / self.n_step, device=x.device)
                t = r + dt
                u = self.model(x, t)
                x = x + u * dt
                traj.append(x.detach().cpu().numpy())

            traj_all.append(traj)
            eps_all.append(x_init.detach().cpu())
            samples_all.append(x.detach().cpu())

        return {
            "sample": torch.cat(samples_all, dim=0),
            "traj": traj_all,
            "eps": torch.cat(eps_all, dim=0),
        }



# ========== Visualization ==========
def visualize_points_and_traj(X0, X1, outputs, title="Flow Trajectories"):
    if isinstance(X0, torch.Tensor):
        X0 = X0.detach().cpu().numpy()
    if isinstance(X1, torch.Tensor):
        X1 = X1.detach().cpu().numpy()
    if isinstance(outputs["sample"], torch.Tensor):
        sample = outputs["sample"].detach().cpu().numpy()
    else:
        sample = outputs["sample"]

    traj = outputs["traj"]
    plt.figure(figsize=(6, 6))
    subset = np.linspace(0, len(traj) - 1, 100, dtype=int)

    # ✨ 淡紫色 RGB (180, 170, 255) → hex = "#B4AAFF"
    lavender = "#B4AAFF"

    for idx in range(min(200, X0.shape[0])):
        xs = [traj[t][idx, 0] for t in subset]
        ys = [traj[t][idx, 1] for t in subset]
        plt.plot(xs, ys, linewidth=0.5, alpha=0.2, color=lavender, zorder=1)

    plt.scatter(X0[:, 0], X0[:, 1], s=3, label="Source", alpha=0.3)
    plt.scatter(X1[:, 0], X1[:, 1], s=3, label="Target", alpha=0.3, c="gray")
    plt.scatter(sample[:, 0], sample[:, 1], s=3, label="Flowed", alpha=0.4, c="darkorange")

    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # plt.legend()
    # plt.axis("equal")
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, transparent=True)


# ===========================================
# COT-FM 專屬可視化
# ===========================================
def visualize_cotfm_traj(clusters, outputs, X1, title="COT-FM Sampling Trajectories"):
    plt.figure(figsize=(6, 6))
    lavender = "#B4AAFF"

    # unpack data
    traj_all = outputs["traj"]
    eps = outputs["eps"].detach().cpu().numpy() if isinstance(outputs["eps"], torch.Tensor) else outputs["eps"]
    sample = outputs["sample"].detach().cpu().numpy() if isinstance(outputs["sample"], torch.Tensor) else outputs["sample"]

    # --- visualize each cluster --- #
    for k, traj in enumerate(traj_all):
        subset = np.linspace(0, len(traj) - 1, 50, dtype=int)

        # (2) Flow trajectories (lavender)
        for idx in range(min(100, traj[0].shape[0])):
            xs = [traj[t][idx, 0] for t in subset]
            ys = [traj[t][idx, 1] for t in subset]
            plt.plot(xs, ys, linewidth=0.5, alpha=0.2, color=lavender, zorder=1)
    
    # (1) Source samples
    Xk = eps
    plt.scatter(Xk[:, 0], Xk[:, 1], s=3, c="tab:blue", alpha=0.3, label="Source")
    plt.scatter(X1[:, 0], X1[:, 1], s=3, label="Target", alpha=0.3, c="gray")


    # (4) Flowed samples (red)
    plt.scatter(sample[:, 0], sample[:, 1], s=3, c="darkorange", alpha=0.4, label="Flowed Samples")

    # (3) Cluster mean
    for k, traj in enumerate(traj_all):
        mu_k = clusters[k][0].detach().cpu().numpy().flatten()
        plt.scatter(mu_k[0], mu_k[1], s=30, c="crimson", marker="x", label=f"Cluster {k+1}")


    # ---- plot config ---- #
    plt.axis("equal")
    plt.xticks([]); plt.yticks([])
    # plt.box(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, transparent=True)
    print(f"Saved visualization: {title.replace(' ', '_')}.png")




# ========== Training wrapper ==========
def train_flow(X0, X1, model_type="flow", n_epoch=500, n_step=1):
    dataloader0 = DataLoader(TensorDataset(X0), batch_size=512, shuffle=True)
    dataloader1 = DataLoader(TensorDataset(X1), batch_size=512, shuffle=True)
    if model_type == "flow":
        model = MLP().to(device)
        trainer = FMTrainer(model, device)
        trainer.fit(dataloader0, dataloader1, n_epoch)
        pipe = FMPipeline(model=model, device=device, n_step=n_step)
        noise = None
    elif model_type == "meanflow":
        model = MeanFlow().to(device)
        trainer = MFTrainer(model, device)
        trainer.fit(dataloader0, dataloader1, n_epoch)
        pipe = MFPipeline(model=model, device=device, n_step=n_step)
        noise = None
    elif model_type == "otcfm":
        model = MLP().to(device)
        trainer = MBOTFMTrainer(model, device)
        trainer.fit(dataloader0, dataloader1, n_epoch)
        pipe = FMPipeline(model=model, device=device, n_step=n_step)
        noise = None
    elif model_type == "cotfm":
        model = MLP().to(device)
        trainer = FMTrainer(model, device)
        trainer.fit(dataloader0, dataloader1, n_epoch)
        trainer = COTFMTrainer(model, device, n_cluster=5)
        noise = trainer.fit(X1, n_epoch)
        pipe = COTFMPipeline(model=model, device=device, clusters=trainer.mean_vars, n_step=n_step)
    elif model_type == "global_otcfm":
        model = MLP().to(device)
        trainer = GlobalOTFMTrainer(model, device)
        trainer.fit(X0, X1, n_epoch)
        pipe = FMPipeline(model=model, device=device, n_step=n_step)
        noise = None
    return (pipe, noise) if noise is not None else pipe


def set_random_seed(random_seed: int = 0, deterministic: bool = True) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# ========== Main ==========
def main():
    set_random_seed(42)

    n_data = 3000

    # Case 1: 1 Gaussian -> 5 Gaussians
    src = Gaussian(mean=[0,0], sigma=0.6)
    # tgt = GaussianMixture5(spread=4.0, sigma=0.35)
    tgt = GaussianMixture5(spread=4.0, sigma=0.5)
    train_X0 = src.sample(n_data)
    train_X1 = tgt.sample(n_data)

    test_X0 = src.sample(n_data)
    test_X1 = tgt.sample(n_data)

    # # fm_pipe = train_flow(train_X0, train_X1, model_type="flow", n_step=100)
    # # fm_gen = fm_pipe(test_X0)
    # # visualize_points_and_traj(test_X0, test_X1, fm_gen, "Flow Matching: 1→5 Gaussians")
    # # wd = cal_wasserstein_distance(fm_gen["sample"], test_X1)
    # # curvature = cal_discrete_curvature(np.array(fm_gen["traj"]).transpose(1,0,2))
    # # print(f"Wasserstein Distance (Flow Matching): {wd:.4f}")
    # # print(f"Discrete Curvature (Flow Matching): {curvature:.4f}")

    # mf_pipe = train_flow(train_X0, train_X1, model_type="meanflow", n_step=1)
    # mf_gen = mf_pipe(test_X0)
    # visualize_points_and_traj(test_X0, test_X1, mf_gen, "MeanFlow: 1→5 Gaussians")
    # wd = cal_wasserstein_distance(mf_gen["sample"], test_X1)
    # curvature = cal_discrete_curvature(np.array(mf_gen["traj"]).transpose(1,0,2))
    # print(f"Wasserstein Distance (MeanFlow): {wd:.4f}")
    # print(f"Discrete Curvature (MeanFlow): {curvature:.4f}")

    # # otcfm_pipe = train_flow(train_X0, train_X1, model_type="otcfm", n_step=100)
    # # otcfm_gen = otcfm_pipe(test_X0)
    # # visualize_points_and_traj(test_X0, test_X1, otcfm_gen, "OT-CFM: 1→5 Gaussians")
    # # wd = cal_wasserstein_distance(otcfm_gen["sample"], test_X1)
    # # curvature = cal_discrete_curvature(np.array(otcfm_gen["traj"]).transpose(1,0,2))
    # # print(f"Wasserstein Distance (OT-CFM): {wd:.4f}")
    # # print(f"Discrete Curvature (OT-CFM): {curvature:.4f}")

    # cotfm_pipe, noise = train_flow(test_X0, test_X1, model_type="cotfm", n_step=100)
    # outputs = cotfm_pipe.sample(noise, n_per_cluster=600)
    # visualize_cotfm_traj(cotfm_pipe.clusters, outputs, test_X1, title="COT-FM: 1→5 Gaussians")
    # wd = cal_wasserstein_distance(outputs["sample"], test_X1)
    # X = np.array(np.array(outputs["traj"]))
    # curvature = cal_discrete_curvature(X.transpose(2, 0, 1, 3).reshape(-1, X.shape[1], X.shape[-1]))
    # print(f"Wasserstein Distance (COT-FM): {wd:.4f}")
    # print(f"Discrete Curvature (COT-FM): {curvature:.4f}")

    # OT-map
    # src = Gaussian(mean=[0,0], sigma=0.5)
    # tgt = Gaussian(mean=[10,0], sigma=0.5)
    # X0 = src.sample(n_data)
    # X1 = tgt.sample(n_data)
    # model = MLP().to(device)
    # gotfm_trainer = GlobalOTFMTrainer(model, device)
    # gotfm_trainer.fit(X0, X1, n_epoch=1)  # skip training, just need pi
    # pi = gotfm_trainer._compute_global_ot(X0, X1)  # this is your plan
    # visualize_ot_matching(X0, X1, pi, title="OT Matching: Gaussians")
    # visualize_random_matching(X0, X1, title="Random Matching: Gaussians")

    # gotfm_pipe = train_flow(X0, X1, model_type="global_otcfm", n_step=1000)
    # visualize_points_and_traj(X0, X1, gotfm_pipe(X0), "Global OT-CFM: 1→5 Gaussians")


    # # Case 2: 1 Gaussian -> Moon shape
    src = Gaussian(mean=[0,0], sigma=0.5)
    tgt = TwoMoons(noise=0.06, rotate=1.3)

    train_X0 = src.sample(n_data)
    train_X1, _ = tgt.sample_target(n_data)

    test_X0 = src.sample(n_data)
    test_X1, _ = tgt.sample_target(n_data)


    # fm_pipe = train_flow(train_X0, train_X1, model_type="flow", n_step=100)
    # fm_outputs = fm_pipe(test_X0)
    # visualize_points_and_traj(test_X0, test_X1, fm_outputs, "Flow Matching: 1→Moon")
    # wd = cal_wasserstein_distance(fm_outputs["sample"], test_X1)
    # curvature = cal_discrete_curvature(np.array(fm_outputs["traj"]).transpose(1,0,2))
    # print(f"Wasserstein Distance (Flow Matching): {wd:.4f}")
    # print(f"Discrete Curvature (Flow Matching): {curvature:.4f}")


    # mf_pipe = train_flow(train_X0, train_X1, model_type="meanflow", n_step=2)
    # mf_outputs = mf_pipe(test_X0)
    # visualize_points_and_traj(test_X0, test_X1, mf_outputs, "MeanFlow: 1→Moon")
    # wd = cal_wasserstein_distance(mf_outputs["sample"], test_X1)
    # curvature = cal_discrete_curvature(np.array(mf_outputs["traj"]).transpose(1,0,2))
    # print(f"Wasserstein Distance (MeanFlow): {wd:.4f}")
    # print(f"Discrete Curvature (MeanFlow): {curvature:.4f}")


    # otcfm_pipe = train_flow(train_X0, train_X1, model_type="otcfm", n_step=100)
    # otcfm_outputs = otcfm_pipe(test_X0)
    # visualize_points_and_traj(test_X0, test_X1, otcfm_outputs, "OT-CFM: 1→Moon")
    # wd = cal_wasserstein_distance(otcfm_outputs["sample"], test_X1)
    # curvature = cal_discrete_curvature(np.array(otcfm_outputs["traj"]).transpose(1,0,2))
    # print(f"Wasserstein Distance (OT-CFM): {wd:.4f}")
    # print(f"Discrete Curvature (OT-CFM): {curvature:.4f}")


    # cotfm_pipe, noise = train_flow(train_X0, train_X1, model_type="cotfm", n_step=100)
    # outputs = cotfm_pipe.sample(noise, n_per_cluster=600)
    # visualize_cotfm_traj(cotfm_pipe.clusters, outputs, test_X1, title="COT-FM: 1→Moon")
    # wd = cal_wasserstein_distance(outputs["sample"], test_X1)
    # X = np.array(np.array(outputs["traj"]))
    # curvature = cal_discrete_curvature(X.transpose(2, 0, 1, 3).reshape(-1, X.shape[1], X.shape[-1]))
    # print(f"Wasserstein Distance (COT-FM): {wd:.4f}")
    # print(f"Discrete Curvature (COT-FM): {curvature:.4f}")




    # Case 3: 1 Gaussian -> 2 Gaussians
    
    n_epoch = 1000
    src = Gaussian(mean=[0,0], sigma=0.6)
    # tgt = GaussianMixture2(sigma=0.5)
    tgt = CheckerboardGrid(
        grid_size=3,
        cell_size=4.0,
        include_white=True,
        # center=(10.0, .0)
        center=(.0, .0)
    )

    n_data = 10000
    train_X0 = src.sample(n_data)
    train_X1 = tgt.sample(n_data)

    n_data = 5000
    test_X0 = src.sample(n_data)
    test_X1 = tgt.sample(n_data)

    # fm_pipe = train_flow(train_X0, train_X1, model_type="flow", n_step=50, n_epoch=n_epoch)
    # fm_gen = fm_pipe(test_X0)
    # visualize_points_and_traj(test_X0, test_X1, fm_gen, "Flow Matching: GaussianGrid")
    # wd = cal_wasserstein_distance(fm_gen["sample"], test_X1)
    # curvature = cal_discrete_curvature(np.array(fm_gen["traj"]).transpose(1,0,2))
    # print(f"Wasserstein Distance (Flow Matching): {wd:.4f}")
    # print(f"Discrete Curvature (Flow Matching): {curvature:.4f}")

    mf_pipe = train_flow(train_X0, train_X1, model_type="meanflow", n_step=1, n_epoch=n_epoch)
    mf_gen = mf_pipe(test_X0)
    visualize_points_and_traj(test_X0, test_X1, mf_gen, "MeanFlow: GaussianGrid")
    wd = cal_wasserstein_distance(mf_gen["sample"], test_X1)
    curvature = cal_discrete_curvature(np.array(mf_gen["traj"]).transpose(1,0,2))
    print(f"Wasserstein Distance (MeanFlow): {wd:.4f}")
    print(f"Discrete Curvature (MeanFlow): {curvature:.4f}")

    # otcfm_pipe = train_flow(train_X0, train_X1, model_type="otcfm", n_step=50, n_epoch=n_epoch)
    # otcfm_gen = otcfm_pipe(test_X0)
    # visualize_points_and_traj(test_X0, test_X1, otcfm_gen, "OT-CFM: GaussianGrid")
    # wd = cal_wasserstein_distance(otcfm_gen["sample"], test_X1)
    # curvature = cal_discrete_curvature(np.array(otcfm_gen["traj"]).transpose(1,0,2))
    # print(f"Wasserstein Distance (OT-CFM): {wd:.4f}")
    # print(f"Discrete Curvature (OT-CFM): {curvature:.4f}")

    # cotfm_pipe, noise = train_flow(train_X0, train_X1, model_type="cotfm", n_step=50, n_epoch=n_epoch)
    # outputs = cotfm_pipe.sample(noise, n_per_cluster=1000)
    # visualize_cotfm_traj(cotfm_pipe.clusters, outputs, test_X1, title="COT-FM: GaussianGrid")
    # wd = cal_wasserstein_distance(outputs["sample"], test_X1)
    # X = np.array(np.array(outputs["traj"]))
    # curvature = cal_discrete_curvature(X.transpose(2, 0, 1, 3).reshape(-1, X.shape[1], X.shape[-1]))
    # print(f"Wasserstein Distance (COT-FM): {wd:.4f}")
    # print(f"Discrete Curvature (COT-FM): {curvature:.4f}")


if __name__ == "__main__":
    main()
