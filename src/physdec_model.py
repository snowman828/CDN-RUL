"""CDN-RUL core model.

Condition-decoupled normalization with conformal uncertainty quantification for
remaining useful life prediction. Three components, corresponding to Section 3
of the manuscript:

  1. CDFE -- Condition-Decoupled Feature Encoder.
     The degradation branch keeps the FULL sensor signal; a gate driven by the
     operating settings softly selects which sensor components feed the
     condition branch z_c, which supplies global operating context to the
     decoder. Sensor streams are normalized per operating regime (Gaussian-
     mixture soft membership, or within-cluster z-scoring); that input-level
     normalization is what renders the degradation stream condition-invariant.

  2. PGDD -- physics-inspired monotonicity prior.
     Degradation is irreversible, so a predicted RUL trajectory is penalized for
     increasing within a window:

         L_phys = mean(ReLU(mu(t) - mu(t+1)))

     This is a soft structural regularizer, not an accuracy driver: the ablation
     in Table 2 places its accuracy effect within seed-to-seed noise.

  3. Decoder and uncertainty.
     LSTM(last-step hidden concatenated with the global condition context) ->
     single mu head, MSE loss. MC-Dropout at inference supplies the epistemic
     uncertainty, which split conformal prediction calibrates -- see
     src/conformal_eval.py.

Ablation switches: use_physics / use_orthogonal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# ① Condition-Decoupled Feature Encoder (CDFE)
# --------------------------------------------------------------------------
class CDFE(nn.Module):
    """Degradation branch keeps the FULL sensor signal; sensors are normalized
    per operating-regime cluster. Normalization mode:
      - 'hard': nearest-cluster assignment (appropriate for DISCRETE regimes —
        FD002/FD004 op-settings snap to 6 points; gained RMSE −12.6% on FD004)
      - 'soft': GMM posterior blending — smooth mixture of per-cluster
        z-scores, avoids pseudo-boundaries on CONTINUOUS regimes (N-CMAPSS DS02
        alt/Mach/TRA/T2 vary continuously; hard binning underperforms there).
    Gate input is op-settings only: membership into the gate is
    redundant with the op-settings themselves)."""

    def __init__(self, n_sensors: int, n_conditions: int = 3,
                 cluster_centers: torch.Tensor | None = None,
                 cluster_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
                 norm_mode: str = "hard",
                 gmm_params: dict | None = None,
                 g_hidden: int = 16, gate_temperature: float = 1.0):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_conditions = n_conditions
        self.T = gate_temperature
        self.norm_mode = norm_mode
        self.n_clusters = 0 if cluster_centers is None else cluster_centers.shape[0]
        if cluster_centers is not None:
            self.register_buffer("cluster_centers", cluster_centers.float())  # (K, C)
            c_mean, c_std = cluster_stats
            self.register_buffer("c_mean", c_mean.float())                    # (K, S)
            self.register_buffer("c_std", c_std.float().clamp_min(1e-6))      # (K, S)
        else:
            self.register_buffer("cluster_centers", torch.zeros(0, n_conditions))
            self.register_buffer("c_mean", torch.zeros(0, n_sensors))
            self.register_buffer("c_std", torch.ones(0, n_sensors))
        if self.norm_mode == "soft" and gmm_params is not None:
            # GMM posterior weights: π_k * N(cond | μ_k, Σ_k)
            self.register_buffer("gmm_means", gmm_params["means"].float())    # (K, C)
            self.register_buffer("gmm_logpi", gmm_params["logpi"].float())    # (K,)
            precs = torch.linalg.inv(gmm_params["covs"].float())              # (K, C, C)
            self.register_buffer("gmm_prec", precs)
            self.register_buffer("gmm_logdet",
                                 gmm_params["covs"].float().logdet())         # (K,)
        # gate: op-settings -> per-sensor soft-selection weights
        self.gate = nn.Sequential(
            nn.Linear(n_conditions, g_hidden), nn.ReLU(),
            nn.Linear(g_hidden, n_sensors),
        )
        # condition branch encoder (softly-selected sensor components)
        self.enc_c = nn.Sequential(nn.Linear(n_sensors, 16), nn.ReLU(),
                                   nn.Linear(16, 16))

    def _posterior(self, cond: torch.Tensor) -> torch.Tensor:
        """GMM posterior over clusters for each timestep: (B, T, K)."""
        K = self.n_clusters
        log_p = torch.empty(cond.shape[:-1] + (K,), device=cond.device)
        const = -0.5 * self.n_conditions * torch.log(torch.tensor(2.0 * torch.pi,
                                                                  device=cond.device))
        for k in range(K):
            diff = cond - self.gmm_means[k]                    # (B, T, C)
            maha = torch.einsum("btc,cd,btd->bt", diff, self.gmm_prec[k], diff)
            log_p[..., k] = self.gmm_logpi[k] + const - 0.5 * self.gmm_logdet[k] \
                - 0.5 * maha
        return torch.softmax(log_p, dim=-1)

    def _normalize(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Per-condition normalization of the degradation stream."""
        if self.n_clusters == 0:
            return x
        if self.norm_mode == "soft":
            w = self._posterior(cond)                          # (B, T, K)
            out = torch.zeros_like(x)
            for k in range(self.n_clusters):
                z = (x - self.c_mean[k]) / self.c_std[k]
                out = out + w[..., k:k + 1] * z
            return out
        # hard: nearest cluster center
        d2 = torch.cdist(cond, self.cluster_centers.unsqueeze(0))  # (B, T, K)
        labels = d2.argmin(dim=-1)                                  # (B, T)
        mean = F.embedding(labels, self.c_mean)                     # (B, T, S)
        std = F.embedding(labels, self.c_std)
        return (x - mean) / std

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """x: (B, T, S) sensors · cond: (B, T, C) op-settings
        Returns z_d (= normalized full x), z_c (B, T, 16), gate mean."""
        x_n = self._normalize(x, cond)
        g = torch.sigmoid(self.gate(cond) / self.T)      # (B, T, S)
        x_c = x_n * g                                    # soft selection
        z_c = self.enc_c(x_c)
        return x_n, z_c, g.mean(dim=(0, 1)).detach()


def orthogonality_loss(z_d: torch.Tensor, z_c: torch.Tensor,
                       proj_c: nn.Linear) -> torch.Tensor:
    """⟨z_d, z_c⟩ normalized per time step (z_c projected to sensor dims)."""
    zc = proj_c(z_c)
    num = (z_d * zc).sum(dim=-1)
    den = z_d.norm(dim=-1) * zc.norm(dim=-1) + 1e-8
    return num.div(den).abs().mean()


# --------------------------------------------------------------------------
# ② Physics-guided degradation dynamics — monotone RUL prior
# --------------------------------------------------------------------------
def monotonicity_loss(step_mu: torch.Tensor) -> torch.Tensor:
    """RUL must be non-increasing in time (degradation is irreversible):
    penalize μ(t) < μ(t+1) (an increase), softly."""
    diff = step_mu[:, :-1] - step_mu[:, 1:]      # >0 = decreasing (good)
    return F.relu(-diff).mean()


# --------------------------------------------------------------------------
# ③ Decoder: LSTM + condition context + single μ head (MSE)
# --------------------------------------------------------------------------
class TemporalEncoder(nn.Module):
    """LSTM decoder; global condition context injected at the last step.

    Returns per-step predictions (for the monotonicity physics term) and the
    final-step prediction (evaluation point)."""

    def __init__(self, n_in: int, n_ctx: int = 16, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers, batch_first=True,
                            dropout=dropout)
        self.fusion = nn.Sequential(nn.Linear(hidden + n_ctx, 64), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z_d: torch.Tensor, z_c: torch.Tensor):
        """z_d (B, T, S) full-signal stream, z_c (B, T, 16) context.
        Returns step_mu (B, T), mu (B,) final-step prediction."""
        out, _ = self.lstm(z_d)                          # (B, T, H)
        ctx = z_c.mean(dim=1)                            # (B, 16)
        ctx_e = ctx.unsqueeze(1).expand(-1, out.size(1), -1)
        h = torch.cat([out, ctx_e], dim=-1)              # (B, T, H+16)
        step_mu = self.head(self.fusion(h)).squeeze(-1)  # (B, T)
        return step_mu, step_mu[:, -1]


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class PhysDecRUL(nn.Module):
    def __init__(self, n_sensors: int, n_conditions: int = 3,
                 use_physics: bool = True, use_orthogonal: bool = True,
                 window: int = 50, cluster_centers: torch.Tensor | None = None,
                 cluster_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
                 norm_mode: str = "hard",
                 gmm_params: dict | None = None):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_conditions = n_conditions
        self.use_physics = use_physics
        self.use_orthogonal = use_orthogonal
        self.norm_mode = norm_mode

        self.cdfe = CDFE(n_sensors, n_conditions,
                         cluster_centers=cluster_centers,
                         cluster_stats=cluster_stats,
                         norm_mode=norm_mode,
                         gmm_params=gmm_params)
        # project condition repr to sensor dims for orthogonality only
        self.proj_c = nn.Linear(16, n_sensors)
        self.encoder = TemporalEncoder(n_in=n_sensors, n_ctx=16)
        self.receptive_field = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """x (B, T, S) sensors · cond (B, T, C) → step_mu, mu, aux."""
        z_d, z_c, gate_mean = self.cdfe(x, cond)
        step_mu, mu = self.encoder(z_d, z_c)
        return step_mu, mu, {"z_d": z_d, "z_c": z_c, "gate": gate_mean}

    def total_loss(self, y: torch.Tensor, step_mu: torch.Tensor, mu: torch.Tensor,
                   z_d: torch.Tensor, z_c: torch.Tensor,
                   lambda_phys: float = 1.0, lambda_orth: float = 1.0) -> dict:
        """MSE (final step, matching reference protocol) + optional terms.
        y (B,), mu (B,) — final-step prediction is the evaluation point."""
        l_mse = F.mse_loss(mu, y)

        l_phys = monotonicity_loss(step_mu) * lambda_phys if self.use_physics \
            else torch.zeros((), device=mu.device)
        l_orth = orthogonality_loss(z_d, z_c, self.proj_c) * lambda_orth \
            if self.use_orthogonal else torch.zeros((), device=mu.device)

        total = l_mse + l_phys + l_orth
        return {"total": total, "mse": l_mse, "phys": l_phys, "orth": l_orth}

    def predict(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Point estimate: final-step prediction (B,)."""
        _, mu, _ = self(x, cond)
        return mu

    def mc_predict(self, x: torch.Tensor, cond: torch.Tensor,
                   n_samples: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-Dropout: (mean, std) of the final-step prediction (B,)."""
        samples = []
        self.train()   # keep dropout active
        with torch.no_grad():
            for _ in range(n_samples):
                _, mu, _ = self(x, cond)
                samples.append(mu)
        s = torch.stack(samples)
        return s.mean(0), s.std(0)
