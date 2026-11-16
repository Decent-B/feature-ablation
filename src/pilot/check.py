"""Step 0: pre-pilot truth check that decides the pilot model.

Measures how well Qwen/Qwen3-1.7B answers "True or False" on Geometry-of-Truth
`cities.csv`, applies the preregistered rule in config/check.yaml and writes
runs/check/decision.json (or decision_smoke.json for --smoke). Only the check model is
ever downloaded or run here; a FAIL is itself the decision to fall back.

Exit codes: 0 PASS/FAIL decided, 2 refused (changed rules without --force), 3 INVALID.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from . import stats
from .utils import (CONFIG_DIR, DATA_CACHE, LOG, MODEL_PATTERNS, RUNS_DIR, fetch_url, hf_snapshot,
                    read_json, sha256_bytes, sha256_file, versions, write_json)

CHECK_DIR = RUNS_DIR / "check"
EXIT_OK, EXIT_REFUSED, EXIT_INVALID = 0, 2, 3


def check_config_path() -> Path:
    return CONFIG_DIR / "check.yaml"


def rule_hash() -> str:
    return sha256_bytes(check_config_path().read_bytes())


def load_check_config() -> dict:
    import yaml

    return yaml.safe_load(check_config_path().read_text())


def _suffix(mode) -> str:
    # mode: False = real check, True = --smoke, "dev" = tiny random model (plumbing test only)
    return "_dev" if mode == "dev" else ("_smoke" if mode is True else "")


def decision_path(smoke) -> Path:
    return CHECK_DIR / f"decision{_suffix(smoke)}.json"


def report_path(smoke) -> Path:
    return CHECK_DIR / f"check_report{_suffix(smoke)}.md"


def load_cities(cfg: dict) -> tuple[list[str], np.ndarray, Path]:
    path = fetch_url(cfg["dataset"]["url"], DATA_CACHE / "truth" / "cities.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    statements = [r["statement"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    return statements, labels, path


def _invalid(reasons: list[str], base: dict, smoke: bool) -> int:
    base.update({"status": "INVALID", "pilot_model": None, "invalid_reasons": reasons})
    write_json(decision_path(smoke), base)
    _write_report(base, smoke)
    LOG.error("CHECK INVALID (harness problem, not a weak model; no fallback): %s", "; ".join(reasons))
    return EXIT_INVALID


def run_check(smoke: bool = False, force: bool = False, dev: bool = False) -> int:
    import torch

    from . import models as M
    from . import prompts as P

    t0 = time.time()
    cfg = load_check_config()
    rhash = rule_hash()
    if dev:
        from .devtiny import ensure_tiny_model

        cfg["model"] = str(ensure_tiny_model())
        smoke = "dev"
        LOG.warning("DEV tiny model check: writes decision_dev.json only; format check relaxed")
    dpath = decision_path(smoke)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    override = None
    if smoke is not True and dpath.exists():
        prev = read_json(dpath)
        if prev.get("rule_hash") != rhash:
            if not force:
                LOG.error("config/check.yaml changed since the existing decision (hash %s -> %s). "
                          "The decision is preregistered; rerun with --force to override (recorded).",
                          str(prev.get("rule_hash"))[:12], rhash[:12])
                return EXIT_REFUSED
            override = {"forced": True, "previous_rule_hash": prev.get("rule_hash"),
                        "previous_status": prev.get("status"), "previous_pilot_model": prev.get("pilot_model"),
                        "previous_decided_at": prev.get("decided_at"),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            LOG.warning("--force: overriding a decision made under different rules; recorded in decision.json")
        else:
            LOG.info("re-running the check under unchanged rules (hash %s)", rhash[:12])

    base = {"smoke": smoke, "rule_hash": rhash, "rule_file": "config/check.yaml",
            "check_model": cfg["model"], "thresholds": cfg["thresholds"],
            "mapping": {"PASS": cfg["on_pass"], "FAIL": cfg["on_fail"]},
            "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "versions": versions(),
            "override": override}

    # ---------------------------------------------------------------- data
    statements, labels, path = load_cities(cfg)
    base["dataset"] = {"path": str(path), "sha256": sha256_file(path), "rows": len(statements),
                       "url": cfg["dataset"]["url"]}
    if len(statements) != int(cfg["dataset"]["expected_rows"]):
        return _invalid([f"dataset loaded {len(statements)} rows, expected {cfg['dataset']['expected_rows']}"],
                        base, smoke)
    if smoke:
        rng = np.random.default_rng(cfg["seed"])
        sel = np.sort(rng.choice(len(statements), size=int(cfg["smoke_n"]), replace=False))
        statements = [statements[i] for i in sel]
        labels = labels[sel]
    n = len(statements)

    # ---------------------------------------------------------------- model
    device = M.pick_device(allow_cpu=bool(smoke))
    dtype = M.default_dtype(device)
    if not dev:
        hf_snapshot(cfg["model"], allow_patterns=MODEL_PATTERNS)
    tok = M.load_tokenizer(cfg["model"])
    model = M.load_model(cfg["model"], device, dtype)
    base["device"] = {"type": device.type, "dtype": str(dtype),
                      "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    tmpl = cfg["template"]
    suffix = P.template_suffix(tok)
    if not suffix["has_empty_think_block"]:
        LOG.warning("chat template suffix has no empty think block: %r", suffix["suffix_text"])
    ids = [P.encode_checked(tok, tmpl.format(statement=s), suffix["suffix_ids"]) for s in statements]

    # format check: greedy decode, response must start with True/False, not a thinking block
    fc = cfg["format_check"]
    fb = M.make_batches(ids[: fc["n"]], tok.pad_token_id, device, fc["n"], 1 << 30)
    outs = {}
    for b in fb:
        for i, o in zip(b.idx, M.generate_greedy(model, tok, b, fc["max_new_tokens"])):
            outs[int(i)] = M.decode(tok, o)
    texts = [outs[i] for i in range(min(fc["n"], n))]
    fmt_ok = [P.starts_with_answer(t) for t in texts]
    base["format_check"] = {"n": len(texts), "passed": int(sum(fmt_ok)), "responses": texts}
    LOG.info("format check: %d/%d responses start with True/False", sum(fmt_ok), len(texts))
    variants = P.answer_variants(tok)
    base["answer_variants"] = variants
    reasons = []
    if not all(fmt_ok) and dev:
        LOG.warning("DEV tiny model: format check failed as expected for random weights; relaxed")
    elif not all(fmt_ok):
        reasons.append(f"format check failed: {len(texts) - sum(fmt_ok)} of {len(texts)} responses "
                       "do not start with True/False (thinking block or other text)")
    for name, v in variants.items():
        if not v["distinct"]:
            reasons.append(f"answer tokens identical for variant '{name}': {v['tokens']}")
    if reasons:
        return _invalid(reasons, base, smoke)

    # ---------------------------------------------------------------- one forward per statement
    tok_ids = {k: v["ids"] for k, v in variants.items()}
    lp = np.zeros((n, 4), dtype=np.float64)
    gather = torch.tensor(tok_ids["bare"] + tok_ids["space"], device=device)
    tf = time.time()
    with torch.inference_mode():
        for b in M.make_batches(ids, tok.pad_token_id, device, int(cfg["batch_size"]), 1 << 30):
            logits = M.last_logits(model, b)
            logp = torch.log_softmax(logits, dim=-1)[:, gather]
            lp[b.idx] = logp.double().cpu().numpy()
    fwd_s = time.time() - tf
    mass = {"bare": np.exp(lp[:, 0]) + np.exp(lp[:, 1]), "space": np.exp(lp[:, 2]) + np.exp(lp[:, 3])}
    chosen = "bare" if mass["bare"].mean() >= mass["space"].mean() else "space"
    off = 0 if chosen == "bare" else 2
    m = lp[:, off] - lp[:, off + 1]          # logit(True) - logit(False)
    coh = mass[chosen]
    LOG.info("answer-token mass: bare %s=%.4f, space %s=%.4f -> using '%s'",
             variants["bare"]["tokens"], mass["bare"].mean(), variants["space"]["tokens"],
             mass["space"].mean(), chosen)

    # ---------------------------------------------------------------- metrics + bootstrap
    au = stats.auroc(m, labels)
    acc = float(np.mean((m > 0).astype(int) == labels))
    coherence = float(coh.mean())
    rng = np.random.default_rng(cfg["seed"])
    W = stats.bootstrap_counts(rng, n, int(cfg["bootstrap"]))
    au_b = stats.auroc_weighted(m, labels, W)
    acc_b = stats.weighted_mean(((m > 0).astype(int) == labels).astype(float), W)
    coh_b = stats.weighted_mean(coh, W)

    def ci(x):
        lo, hi = stats.percentile_ci(x)
        return [float(lo), float(hi)]

    metrics = {
        "auroc": {"value": au, "ci95": ci(au_b)},
        "accuracy_at_0": {"value": acc, "ci95": ci(acc_b)},
        "coherence": {"value": coherence, "ci95": ci(coh_b)},
        "label_balance": float(labels.mean()),
        "n": n,
        "mass_by_variant": {k: float(v.mean()) for k, v in mass.items()},
        "answer_variant": chosen,
        "answer_tokens": variants[chosen]["tokens"],
        "mean_margin_by_label": {"true": float(m[labels == 1].mean()), "false": float(m[labels == 0].mean())},
        "forward_seconds": round(fwd_s, 2),
    }
    thr = cfg["thresholds"]
    failed = []
    if au < thr["auroc"]:
        failed.append(f"auroc {au:.4f} < {thr['auroc']}")
    if acc < thr["accuracy"]:
        failed.append(f"accuracy {acc:.4f} < {thr['accuracy']}")
    if coherence < thr["coherence"]:
        failed.append(f"coherence {coherence:.4f} < {thr['coherence']}")
    status = "PASS" if not failed else "FAIL"
    base.update({"status": status, "pilot_model": cfg["on_pass"] if status == "PASS" else cfg["on_fail"],
                 "failed_conditions": failed, "metrics": metrics,
                 "wall_seconds": round(time.time() - t0, 1)})
    write_json(dpath, base)
    _write_report(base, smoke)
    LOG.info("CHECK %s -> pilot_model = %s%s", status, base["pilot_model"],
             f" (failed: {'; '.join(failed)})" if failed else "")
    LOG.info("wrote %s", dpath)
    return EXIT_OK


def _write_report(d: dict, smoke: bool) -> None:
    L = [f"# Step 0 truth check{' (SMOKE: not a decision)' if smoke else ''}", ""]
    L.append(f"**Status: {d.get('status')}** → pilot model: `{d.get('pilot_model')}`")
    L.append("")
    if d.get("invalid_reasons"):
        L.append("INVALID means a harness problem (no decision, no fallback):")
        L += [f"- {r}" for r in d["invalid_reasons"]]
        L.append("")
    m = d.get("metrics")
    if m:
        thr = d["thresholds"]
        L += ["| Condition | Value | 95% CI | Threshold | Pass |", "|---|---|---|---|---|"]
        for key, tkey, name in (("auroc", "auroc", "AUROC of m(x)"),
                                ("accuracy_at_0", "accuracy", "Accuracy, sign of m(x) at 0"),
                                ("coherence", "coherence", "Coherence P(True)+P(False)")):
            v = m[key]
            L.append(f"| {name} | {v['value']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | "
                     f"≥ {thr[tkey]} | {'yes' if v['value'] >= thr[tkey] else 'no'} |")
        L += ["", f"- n = {m['n']}, label balance (fraction true) = {m['label_balance']:.3f}",
              f"- answer tokens: {m['answer_tokens']} (variant `{m['answer_variant']}`; mean mass by "
              f"variant {m['mass_by_variant']})",
              f"- mean m(x): true {m['mean_margin_by_label']['true']:.3f}, false "
              f"{m['mean_margin_by_label']['false']:.3f}"]
    fc = d.get("format_check")
    if fc:
        L += ["", f"Format check: {fc['passed']}/{fc['n']} greedy responses start with True/False."]
    L += ["", f"- rule file `config/check.yaml`, SHA-256 `{d['rule_hash']}`",
          f"- check model `{d['check_model']}`; mapping PASS→`{d['mapping']['PASS']}`, "
          f"FAIL→`{d['mapping']['FAIL']}`",
          f"- decided at {d['decided_at']}"]
    if d.get("override"):
        L.append(f"- **override (--force)**: {d['override']}")
    report_path(smoke).write_text("\n".join(L) + "\n")


def verify_decision_for_pilot() -> dict:
    """Used by the pilot: the decision must exist, be PASS/FAIL and match the current rules."""
    p = decision_path(False)
    if not p.exists():
        raise FileNotFoundError(str(p))
    d = read_json(p)
    if d.get("status") not in ("PASS", "FAIL"):
        raise SystemExit(f"ERROR: {p} has status {d.get('status')!r}; an INVALID check is a harness bug. "
                         "Fix it and rerun `bash run_pilot.sh --check`.")
    if d.get("rule_hash") != rule_hash():
        raise SystemExit(f"ERROR: config/check.yaml no longer matches the hash in {p}. The decision is "
                         "binding: restore the file or rerun `bash run_pilot.sh --check --force`.")
    if not d.get("pilot_model"):
        raise SystemExit(f"ERROR: {p} names no pilot model")
    d["_decision_file_sha256"] = sha256_file(p)
    return d
