"""Behavioral scores, the refusal classifier, and per-configuration damage / cost.

Truth   m(x) = logit(True) - logit(False) at the first response position.
        damage = AUROC(clean) - AUROC(ablated) on all eval examples.
Refusal r(x) = log P(refusal-start) - log P(everything else) at the first response position.
        damage = refusal rate (r > tau) on the harmful eval prompts the clean model refuses
        (clean generation labels), clean minus ablated. Generation labels replace r > tau if
        the classifier failed calibration.
Cost    mean KL(p_clean || p_ablated) at the first response position on generic prompts
        (primary); mean ||x' - x|| / ||x|| at intervention positions (secondary).
"""

from __future__ import annotations

import numpy as np
import torch

from . import stats
from .models import log1mexp


# --------------------------------------------------------------------------- scores from logits

def truth_scores(logits: torch.Tensor, ids: list[int]):
    lp = torch.log_softmax(logits.float(), dim=-1)
    m = lp[:, ids[0]] - lp[:, ids[1]]
    coh = lp[:, ids[0]].exp() + lp[:, ids[1]].exp()
    return m, coh


def refusal_scores(logits: torch.Tensor, start_ids: list[int]):
    lp = torch.log_softmax(logits.float(), dim=-1)
    lpr = torch.logsumexp(lp[:, start_ids], dim=-1).clamp(max=-1e-7)
    return lpr - log1mexp(lpr), lpr.exp()


def kl_from_logprobs(lp_clean: torch.Tensor, logits_abl: torch.Tensor) -> torch.Tensor:
    lpa = torch.log_softmax(logits_abl.float(), dim=-1)
    return (lp_clean.exp() * (lp_clean - lpa)).sum(-1).clamp_min(0.0)


# --------------------------------------------------------------------------- refusal labels

def is_refusal(text: str, substrings: list[str]) -> bool:
    return any(s in text for s in substrings)


def refusal_start_set(first_tokens: list[int], labels: list[bool], coverage: float) -> list[int]:
    toks = [t for t, l in zip(first_tokens, labels) if l]
    if not toks:
        return []
    vals, counts = np.unique(np.array(toks), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    cum = np.cumsum(counts[order]) / counts.sum()
    n = int(np.searchsorted(cum, coverage - 1e-12) + 1)
    return [int(v) for v in vals[order][:n]]


def calibrate_threshold(r: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold on r maximising agreement with generation labels. Returns (tau, agreement)."""
    s = np.sort(np.unique(r))
    cands = np.concatenate([[s[0] - 1.0], (s[:-1] + s[1:]) / 2, [s[-1] + 1.0]]) if len(s) > 1 else np.array([s[0]])
    best_tau, best_acc = 0.0, -1.0
    for tau in cands:
        acc = float(np.mean((r > tau) == labels))
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(tau) < abs(best_tau)):
            best_tau, best_acc = float(tau), acc
    return best_tau, best_acc


# --------------------------------------------------------------------------- per-config summaries

def truth_point(score, clean, labels, coh) -> dict:
    au_c = stats.auroc(clean, labels)
    au_a = stats.auroc(score, labels)
    correct = (clean > 0).astype(int) == labels
    out = {
        "auroc_clean": au_c, "auroc": au_a, "damage": au_c - au_a,
        "accuracy": float(np.mean((score > 0).astype(int) == labels)),
        "accuracy_clean": float(np.mean(correct)),
        "flip_rate": float(np.mean(np.sign(score) != np.sign(clean))),
        "bias_shift": float(np.mean(score - clean)),
        "bias_shift_true": float(np.mean((score - clean)[labels == 1])),
        "bias_shift_false": float(np.mean((score - clean)[labels == 0])),
        "coherence": float(np.mean(coh)),
    }
    if correct.sum() > 1 and len(np.unique(labels[correct])) == 2:
        out["damage_correct_subset"] = (stats.auroc(clean[correct], labels[correct])
                                        - stats.auroc(score[correct], labels[correct]))
    else:
        out["damage_correct_subset"] = 0.0
    return out


def refusal_point(score, clean, labels, refused_mask, tau, gen_abl=None) -> dict:
    harmful_ref = refused_mask
    if gen_abl is not None:
        rate_abl = float(np.mean(gen_abl)) if len(gen_abl) else 0.0
        rate_clean = 1.0
    else:
        rate_abl = float(np.mean(score[harmful_ref] > tau)) if harmful_ref.any() else 0.0
        rate_clean = float(np.mean(clean[harmful_ref] > tau)) if harmful_ref.any() else 0.0
    harmless = labels == 0
    return {
        "refusal_rate_clean": rate_clean, "refusal_rate": rate_abl, "damage": rate_clean - rate_abl,
        "harmless_refusal_rate": float(np.mean(score[harmless] > tau)) if harmless.any() else 0.0,
        "auroc": stats.auroc(score, labels), "auroc_clean": stats.auroc(clean, labels),
        "bias_shift": float(np.mean(score - clean)),
        "coherence": float("nan"),
    }


def damage_boot(task: str, scores: np.ndarray, clean: np.ndarray, labels: np.ndarray, We: np.ndarray,
                refused_mask=None, tau=None, gen_abl: np.ndarray | None = None) -> np.ndarray:
    """Damage per bootstrap replicate for C configurations. scores [C, n_eval] -> [C, B]."""
    C = scores.shape[0]
    B = We.shape[0]
    out = np.zeros((C, B))
    if task == "truth":
        au_c = stats.auroc_weighted(clean, labels, We)
        for c in range(C):
            out[c] = au_c - stats.auroc_weighted(scores[c], labels, We)
        return out
    Wr = We[:, refused_mask]
    if gen_abl is not None:
        out[:] = 1.0 - stats.weighted_mean(gen_abl.astype(float), Wr)
        return out
    rc = stats.weighted_mean((clean[refused_mask] > tau).astype(float), Wr)
    ra = stats.weighted_mean((scores[:, refused_mask] > tau).astype(float), Wr)
    return rc[None, :] - ra


def bias_boot(scores, clean, We) -> np.ndarray:
    return stats.weighted_mean(scores - clean[None, :], We)
