"""Load HOT3D → YOLOE text prompt vocabulary."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from lib.paths import EGOUSE_ROOT

PROMPTS_PATH = EGOUSE_ROOT / "config" / "hot3d_yoloe_prompts.json"


@lru_cache(maxsize=1)
def load_hot3d_yoloe_prompts(path: Path | None = None) -> dict:
  cfg_path = path or PROMPTS_PATH
  data = json.loads(cfg_path.read_text())
  rows = data["prompts"]
  instance_to_prompt = {r["instance_name"]: r["prompt"] for r in rows}
  prompt_to_instance = {r["prompt"]: r["instance_name"] for r in rows}
  prompt_list = [r["prompt"] for r in rows]
  return {
    "instance_to_prompt": instance_to_prompt,
    "prompt_to_instance": prompt_to_instance,
    "prompt_list": prompt_list,
    "n_prompts": len(prompt_list),
  }
