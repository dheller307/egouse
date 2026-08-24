import sys
from pathlib import Path

EGOUSE_ROOT = Path(__file__).resolve().parent.parent
HOT3D_ROOT = EGOUSE_ROOT / "hot3d_repo" / "hot3d"
DEFAULT_OUTPUT_DIR = EGOUSE_ROOT / "outputs"
HOT3D_DATA_DIR = EGOUSE_ROOT / "hot3d_data"

# Main EgoUse environment (teacher/student CLIs, pytest).
VENV_MAIN = EGOUSE_ROOT / "venv"

# Detector venvs:
#   venv_doh    → 100DOH hand detector (Hand_Object_Detector)
#   venv_yoloe  → YOLOE object detector (ultralytics)
VENV_DOH = EGOUSE_ROOT / "venv_doh"
VENV_YOLOE = EGOUSE_ROOT / "venv_yoloe"


def venv_python(venv_dir: Path) -> Path:
  return venv_dir / "bin" / "python"


def doh_torch_lib() -> Path | None:
  lib = VENV_DOH / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
  return lib if lib.is_dir() else None

# Frozen student checkpoint (phase_d r7).
DEFAULT_STUDENT_CKPT = EGOUSE_ROOT / "checkpoints" / "phase_d_r7" / "best.pt"


def resolve_student_checkpoint() -> Path:
  return DEFAULT_STUDENT_CKPT


def setup_paths():
  for path in (EGOUSE_ROOT, HOT3D_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
      sys.path.insert(0, path_str)
