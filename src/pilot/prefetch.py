"""Background model prefetch started by run_pilot.sh while PyTorch installs.

Uses only the standard library and huggingface_hub (numpy/torch/pyyaml may not be installed
yet). Downloads config, tokenizer and *.safetensors into the shared Hugging Face cache.
  python -m pilot.prefetch check   -> the Step 0 model only (never the fallback)
  python -m pilot.prefetch smoke   -> Qwen/Qwen3-1.7B
  python -m pilot.prefetch pilot   -> the model named in runs/check/decision.json (or the check model)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab.json", "*.jinja"]


def check_model() -> str:
    m = re.search(r"^model:\s*(\S+)", (ROOT / "config" / "check.yaml").read_text(), re.M)
    return m.group(1) if m else "Qwen/Qwen3-1.7B"


def target(kind: str) -> str:
    if kind == "smoke":
        return "Qwen/Qwen3-1.7B"
    if kind == "pilot":
        d = ROOT / "runs" / "check" / "decision.json"
        if d.exists():
            try:
                pm = json.loads(d.read_text()).get("pilot_model")
                if pm:
                    return pm
            except Exception:  # noqa: BLE001
                pass
    return check_model()


def main() -> int:
    from huggingface_hub import snapshot_download

    kind = sys.argv[1] if len(sys.argv) > 1 else "check"
    repo = target(kind)
    print(f"[prefetch] downloading {repo} (config, tokenizer, safetensors)", flush=True)
    path = snapshot_download(repo, allow_patterns=PATTERNS)
    print(f"[prefetch] done: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
