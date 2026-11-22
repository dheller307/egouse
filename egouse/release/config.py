"""Immutable release identity and explicit host-specific asset paths."""

import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'config' / 'medium33-release.json'
ASSET_KEYS = ('student_checkpoint', 'object_checkpoint', 'hand_checkpoint',
              'hand_repository', 'hand_python', 'object_python')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def user_paths(config):
    local = load(config)
    if set(local) != set(ASSET_KEYS):
        raise ValueError(f'Asset config must contain exactly {ASSET_KEYS}')
    return {name: (config.parent / Path(local[name])).absolute() for name in ASSET_KEYS}


def verify_environment(role, expected):
    found = {'python': platform.python_version()}
    for name in expected:
        if name != 'python':
            found[name] = importlib.metadata.version(name)
    if found != expected:
        raise ValueError(f'{role} runtime differs from validated pins: {found}')
    return found


def verify_child(role, python, expected):
    command = [str(python), '-B', '-c',
               "import sys,json; sys.path.insert(0,sys.argv[1]); "
               "from release.config import verify_environment; "
               "print(json.dumps(verify_environment(sys.argv[2],json.loads(sys.argv[3]))))",
               str(ROOT), role, json.dumps(expected)]
    return json.loads(subprocess.check_output(command, text=True))
