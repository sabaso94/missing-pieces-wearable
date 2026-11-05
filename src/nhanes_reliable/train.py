"""Training the deep survival models: inputs per fold, modality and day dropout, early stopping, ensembles."""

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .activity import DayData
from .data import BLOCKS, REMOVABLE
from .metrics import HORIZON, c_index
from .model import FusionSurvivalNet
from .survival import discretise, logistic_hazard_loss, risk_at
from .tabular import block_flags, fit_tab_stats, make_tab_features, tab_variables

DEEP_MODELS: dict[str, dict] = {
    # All blocks; random blocks and days dropped in training, so one model serves every missing-data case.
    "fused": {"blocks": ["core", "wearable", "labs", "exam", "questionnaire"], "use_wearable": True,
              "modality_dropout": True, "day_dropout": True},
    # The same without modality dropout: never sees a block missing on purpose (the naive comparison in E2).
    "fused_naive": {"blocks": ["core", "wearable", "labs", "exam", "questionnaire"], "use_wearable": True,
                    "modality_dropout": False, "day_dropout": True},
    "deep_wearable": {"blocks": ["core", "wearable"], "use_wearable": True,
                      "modality_dropout": False, "day_dropout": True},
    "deep_clinical": {"blocks": ["core", "labs", "exam", "questionnaire"], "use_wearable": False,
                      "modality_dropout": True, "day_dropout": False},
}


@dataclass
class TrainConfig:
    model: str
    seed: int = 0
    device: str = "cuda"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    p_block_drop: float = 0.3


@dataclass
class FoldInputs:
    """Everything one fold's models need, aligned to the cohort rows. Statistics come from the fit rows."""
    x: np.ndarray              # (N, 7, 2, 1440) day tensor (shared, not copied)
    day_mask: np.ndarray       # (N, 7) bool
    tab: np.ndarray            # (N, F) float32 standardised values + missing indicators
    tab_blocks: np.ndarray     # (F,) block of each tabular column
    tab_indicator: np.ndarray  # (F,) True for missing-indicator columns
    flags: np.ndarray          # (N, 4) block-present flags (blocks the model does not use are 0)
    interval: np.ndarray       # (N,) yearly interval of death or censoring
    died: np.ndarray
    time: np.ndarray
    mims_mean: float           # log10(1 + MIMS) mean and std over worn minutes of the fit rows
    mims_std: float
    use_wearable: bool


def build_fold_inputs(cohort: pd.DataFrame, days: DayData, fit_idx: np.ndarray, model: str) -> FoldInputs:
    """Model inputs for one fold; `days` must be aligned to the cohort rows."""
    if not np.array_equal(days.seqn, cohort.index.to_numpy()):
        raise ValueError("days are not aligned to the cohort rows")
    spec = DEEP_MODELS[model]
    variables = tab_variables(spec["blocks"])
    tab, names = make_tab_features(cohort, fit_tab_stats(cohort.iloc[fit_idx], variables), variables)
    block_of = {v: b for b, vs in BLOCKS.items() for v in vs}
    tab_blocks = np.array([block_of[n.removesuffix("_missing")] for n in names])
    tab_indicator = np.array([n.endswith("_missing") for n in names])

    flags = block_flags(cohort)
    for i, block in enumerate(REMOVABLE):
        if block not in spec["blocks"]:
            flags[:, i] = 0.0

    worn = days.x[fit_idx, :, 1] > 0
    worn &= days.day_mask[fit_idx][:, :, None]
    values = days.x[fit_idx, :, 0][worn].astype(np.float64)
    return FoldInputs(
        x=days.x, day_mask=days.day_mask, tab=tab, tab_blocks=tab_blocks, tab_indicator=tab_indicator, flags=flags,
        interval=discretise(cohort["time_years"].to_numpy()), died=cohort["died"].to_numpy(),
        time=cohort["time_years"].to_numpy(), mims_mean=float(values.mean()), mims_std=float(values.std()),
        use_wearable=spec["use_wearable"],
    )


def sample_block_dropout(flags: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    """Turn each present block off with probability p (a missing block never becomes present)."""
    drop = torch.rand(flags.shape, generator=generator) < p
    return flags * (~drop).to(flags.device).float()


def sample_day_dropout(day_mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Keep a random number k in [1, n_valid] of each person's valid days (a random subset of them)."""
    n_valid = day_mask.sum(dim=1)
    k = (torch.rand(len(day_mask), generator=generator) * n_valid.cpu()).floor().long() + 1
    keys = torch.rand(day_mask.shape, generator=generator)
    keys[~day_mask.cpu()] = float("inf")
    rank = keys.argsort(dim=1).argsort(dim=1)
    return day_mask & (rank < k[:, None]).to(day_mask.device)


def apply_flags(tab: torch.Tensor, day_mask: torch.Tensor, flags: torch.Tensor,
                inputs: FoldInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Blank every block whose flag is 0 exactly as a block missing in the data looks: values 0, indicators 1."""
    block_index = torch.tensor([REMOVABLE.index(b) if b in REMOVABLE else -1 for b in inputs.tab_blocks],
                               device=tab.device)
    indicator = torch.from_numpy(inputs.tab_indicator).to(tab.device)
    off = (flags[:, block_index.clamp(min=0)] == 0) & (block_index >= 0)
    tab = torch.where(off & ~indicator, torch.zeros_like(tab), tab)
    tab = torch.where(off & indicator, torch.ones_like(tab), tab)
    return tab, day_mask & (flags[:, :1] > 0)


def _batch(inputs: FoldInputs, idx: np.ndarray, device: str) -> tuple:
    """Tensors for rows idx; log-MIMS standardised with the fit statistics, non-worn minutes stay 0."""
    tab = torch.from_numpy(inputs.tab[idx]).to(device)
    flags = torch.from_numpy(inputs.flags[idx]).to(device)
    mask = torch.from_numpy(inputs.day_mask[idx]).to(device)
    days = None
    if inputs.use_wearable:
        days = torch.from_numpy(inputs.x[idx]).to(device)
        worn = days[:, :, 1:2] > 0
        scaled = (days[:, :, 0:1] - inputs.mims_mean) / inputs.mims_std
        days = torch.cat([torch.where(worn, scaled, torch.zeros_like(scaled)), days[:, :, 1:2]], dim=2)
    return days, mask, tab, flags


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _new_model(inputs: FoldInputs) -> FusionSurvivalNet:
    return FusionSurvivalNet(n_tab=inputs.tab.shape[1], use_wearable=inputs.use_wearable)


@torch.no_grad()
def predict_risk(models: list[FusionSurvivalNet], inputs: FoldInputs, idx: np.ndarray, removed: list[str] = (),
                 day_mask: np.ndarray | None = None, years: int = int(HORIZON), device: str = "cuda",
                 batch_size: int = 512) -> np.ndarray:
    """Ensemble mean of the members' predicted risk of death before `years`, with `removed` blocks blanked."""
    risks = []
    for model in models:
        model.to(device).eval()
        out = []
        for start in range(0, len(idx), batch_size):
            rows = idx[start : start + batch_size]
            days, mask, tab, flags = _batch(inputs, rows, device)
            if day_mask is not None:
                mask = torch.from_numpy(day_mask[rows]).to(device)
            for block in removed:
                flags[:, REMOVABLE.index(block)] = 0.0
            tab, mask = apply_flags(tab, mask, flags, inputs)
            hazards = torch.sigmoid(model(days, mask, tab, flags)).cpu().numpy().astype(np.float64)
            out.append(risk_at(hazards, years))
        risks.append(np.concatenate(out))
    return np.mean(risks, axis=0)


def train_model(cfg: TrainConfig, inputs: FoldInputs, fit_idx: np.ndarray,
                hold_idx: np.ndarray) -> tuple[FusionSurvivalNet, pd.DataFrame]:
    """AdamW on the logistic-hazard loss; keeps the epoch with the best hold-out C-index (5-year risk)."""
    _seed_everything(cfg.seed)
    spec = DEEP_MODELS[cfg.model]
    generator = torch.Generator().manual_seed(cfg.seed)
    model = _new_model(inputs).to(cfg.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_c, best_state, since_best, history = -1.0, None, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        order = fit_idx[torch.randperm(len(fit_idx), generator=generator).numpy()]
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            days, mask, tab, flags = _batch(inputs, rows, cfg.device)
            if spec["modality_dropout"]:
                flags = sample_block_dropout(flags, cfg.p_block_drop, generator)
            if spec["day_dropout"] and inputs.use_wearable:
                mask = sample_day_dropout(mask, generator)
            tab, mask = apply_flags(tab, mask, flags, inputs)
            logits = model(days, mask, tab, flags)
            loss = logistic_hazard_loss(logits, torch.from_numpy(inputs.interval[rows]).to(cfg.device),
                                        torch.from_numpy(inputs.died[rows]).float().to(cfg.device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        risk = predict_risk([model], inputs, hold_idx, device=cfg.device)
        hold_c = c_index(inputs.time[hold_idx], inputs.died[hold_idx], risk)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "hold_c": hold_c})
        if hold_c > best_c:
            best_c, best_state, since_best = hold_c, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history)


def train_ensemble(cfg: TrainConfig, inputs: FoldInputs, fit_idx: np.ndarray, hold_idx: np.ndarray,
                   ckpt_dir: Path, seeds=range(5)) -> list[FusionSurvivalNet]:
    """One model per seed; a member whose checkpoint already exists is loaded instead of trained."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for seed in seeds:
        path = ckpt_dir / f"seed{seed}.pt"
        if path.exists():
            model = _new_model(inputs)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.to(cfg.device).eval()
        else:
            model, history = train_model(TrainConfig(**{**cfg.__dict__, "seed": seed}), inputs, fit_idx, hold_idx)
            torch.save(model.state_dict(), path)
            history.to_csv(ckpt_dir / f"seed{seed}_history.csv", index=False)
        models.append(model)
    return models
