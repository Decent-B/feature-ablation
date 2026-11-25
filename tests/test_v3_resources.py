"""Competing memory use (resources_v3): wait for free memory, fit batches to it, back off and retry on a memory stop."""

import pytest
import torch


class _WD:
    def __init__(self, thr=16.0):
        self.threshold_gb, self.tripped, self.trip_reason = thr, False, ""


class _MC:
    def __init__(self, batch_mem=40.0, weights=5.0, loaded=True):
        self.tuning = {"batch_mem_gb": batch_mem, "weights_gb": weights}
        self._model = object() if loaded else None
        self._prefix, self._sae = {"t": 1}, {"s": 1}


class _Run:
    def __init__(self, tmp, **mem):
        self.run_dir = tmp
        self.cfg = {"memory": {"safety_gb": 8, "max_wait_minutes": 1, "poll_seconds": 30, "max_backoffs": 3,
                               "min_batch_factor": 0.0625, **mem}}
        self.wd = _WD()
        self._m = {"g": _MC()}


def _res(run, avail):
    from pilot.resources_v3 import Resources

    seq = list(avail) if isinstance(avail, (list, tuple)) else None
    fn = (lambda: seq.pop(0) if len(seq) > 1 else seq[0]) if seq else (lambda: float(avail))
    run.res = Resources(run, mem_fn=fn, sleep_fn=lambda s: None, total_gb=128.0)
    return run.res


def test_fit_halves_until_batches_fit(tmp_path):
    run = _Run(tmp_path)
    r = _res(run, 16 + 8 + 25)                     # 25 GB headroom, batches need 40 GB at x1
    r.prepare("s")
    assert r.factor == 0.5
    r2 = _res(run, 200)                            # plenty free: back to the cap (1)
    r2.prepare("s")
    assert r2.factor == 1.0


def test_waits_for_memory_then_runs(tmp_path):
    run = _Run(tmp_path)
    r = _res(run, [10, 10, 100])
    r.prepare("s")
    kinds = [e["kind"] for e in r.state["events"]]
    assert kinds[0] == "wait" and "resumed" in kinds


def test_gives_up_when_memory_never_frees(tmp_path):
    from pilot.resources_v3 import ResourceError

    run = _Run(tmp_path)
    r = _res(run, 10)
    with pytest.raises(ResourceError):
        r.prepare("s")
    assert r.state["events"][-1]["kind"] == "gave_up"


def test_backoff_on_oom_then_succeeds(tmp_path):
    from pilot.resources_v3 import run_with_backoff

    run = _Run(tmp_path)
    _res(run, 200)
    calls = []

    def fn():
        calls.append(run.res.factor)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return 7

    assert run_with_backoff(run, "ablate", fn) == 7
    assert calls == [1.0, 0.5]
    assert run.res.state["cap"] == 0.5 and run._m["g"]._prefix == {} and run._m["g"]._sae == {}


def test_watchdog_stop_is_retried_and_rearmed(tmp_path):
    from pilot.resources_v3 import run_with_backoff

    run = _Run(tmp_path)
    _res(run, 200)
    n = []

    def fn():
        n.append(1)
        if len(n) == 1:
            run.wd.tripped, run.wd.trip_reason = True, "MemAvailable fell below 16 GB"
            raise KeyboardInterrupt
        assert run.wd.tripped is False             # re-armed before the retry
        return "ok"

    assert run_with_backoff(run, "prep", fn) == "ok"
    assert [e["kind"] for e in run.res.state["events"]].count("backoff") == 1


def test_real_ctrl_c_is_not_retried(tmp_path):
    from pilot.resources_v3 import run_with_backoff

    run = _Run(tmp_path)
    _res(run, 200)

    def fn():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_with_backoff(run, "prep", fn)


def test_other_errors_are_not_retried(tmp_path):
    from pilot.resources_v3 import run_with_backoff

    run = _Run(tmp_path)
    _res(run, 200)
    with pytest.raises(ValueError):
        run_with_backoff(run, "prep", lambda: (_ for _ in ()).throw(ValueError("bug")))


def test_gives_up_after_max_backoffs_and_state_persists(tmp_path):
    from pilot.resources_v3 import ResourceError, Resources, run_with_backoff

    run = _Run(tmp_path, max_backoffs=2)
    _res(run, 200)

    def fn():
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(ResourceError):
        run_with_backoff(run, "ablate", fn)
    assert run.res.state["cap"] == 0.25
    again = Resources(run, mem_fn=lambda: 200.0, sleep_fn=lambda s: None, total_gb=128.0)
    assert again.state["cap"] == 0.25                # a resumed run keeps the smaller batches


def test_unloaded_weights_count_toward_the_start_threshold(tmp_path):
    run = _Run(tmp_path)
    run._m = {"g": _MC(weights=10.0, loaded=False)}
    r = _res(run, [16 + 8 + 5, 16 + 8 + 20])        # 29 GB < 16 + 8 + 10 -> waits once
    r.prepare("s")
    assert r.state["events"][0]["kind"] == "wait"
