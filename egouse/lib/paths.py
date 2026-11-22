import os
import sys
from pathlib import Path

EGOUSE_ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = Path(os.environ.get("EGOUSE_ASSET_ROOT", EGOUSE_ROOT)).resolve()
HOT3D_ROOT = ASSET_ROOT / "hot3d_repo" / "hot3d"
DEFAULT_OUTPUT_DIR = EGOUSE_ROOT / "outputs"
HOT3D_DATA_DIR = ASSET_ROOT / "hot3d_data"

# Main EgoUse environment (teacher/student CLIs, pytest).
VENV_MAIN = ASSET_ROOT / "venv"

# Detector venvs:
#   venv_doh    → 100DOH hand detector (Hand_Object_Detector)
#   venv_yoloe  → YOLOE object detector (ultralytics)
VENV_DOH = ASSET_ROOT / "venv_doh"
VENV_YOLOE = ASSET_ROOT / "venv_yoloe"


def venv_python(venv_dir: Path) -> Path:
  return venv_dir / "bin" / "python"


def doh_torch_lib() -> Path | None:
  lib = VENV_DOH / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
  return lib if lib.is_dir() else None

# Versioned student checkpoint (phase_d r7, ~1.3 MB).
DEFAULT_STUDENT_CKPT = EGOUSE_ROOT / "checkpoints" / "phase_d_r7" / "best.pt"
LEGACY_STUDENT_CKPT = (
  EGOUSE_ROOT
  / "outputs"
  / "student"
  / "gru_loop"
  / "explore_chain_phase_d_50r"
  / "round_7"
  / "best.pt"
)


def resolve_student_checkpoint() -> Path:
  """Prefer committed checkpoints/ path; fall back to local outputs/ tree."""
  if DEFAULT_STUDENT_CKPT.is_file():
    return DEFAULT_STUDENT_CKPT
  if LEGACY_STUDENT_CKPT.is_file():
    return LEGACY_STUDENT_CKPT
  return DEFAULT_STUDENT_CKPT


def setup_paths():
  for path in (EGOUSE_ROOT, HOT3D_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
      sys.path.insert(0, path_str)
