"""Capture one fresh release invocation or inspect existing stage receipts."""

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from release.config import MANIFEST,load,sha


def known_hardware():
    cpu=platform.processor()
    if Path('/proc/cpuinfo').is_file():
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')),cpu)
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],
                                    text=True,stderr=subprocess.DEVNULL,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):
        gpu=None
    return dict(cpu=cpu,gpu=gpu,platform=platform.platform())


def inspect(run):
    provenance=load(run/'provenance.json')
    receipt={}
    for name in ('pipeline','student','hands','objects'):
        file=run/f'{name}-runtime.json'
        if file.is_file():
            receipt[name]=load(file)
    if 'student' not in receipt:
        raise ValueError('Missing Student runtime receipt')
    frames=receipt['student']['frames']
    pipeline=receipt.get('pipeline',{})
    return dict(versions=provenance.get('environments'),frames=frames,resolution=pipeline.get('resolution'),
         threads=pipeline.get('threads',load(MANIFEST)['threads']) if provenance.get('release_sha256')==sha(MANIFEST) else pipeline.get('threads'),
         receipts=receipt,
         observed_vrs_batch_fps=frames/pipeline['end_to_end_s'] if pipeline else None,
         detector_stage_inference_fps={name:frames/r['inference_s'] for name,r in receipt.items() if name in ('hands','objects')},
         startup_conditions='Fresh invocation per capture; code reports load and inference separately. OS cache state is uncontrolled.',
         scope='Offline batch processing. Stage-only rates exclude other stages; no live latency or sustained full-pipeline measurement.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Existing receipt directory or output for capture')
    p.add_argument('--metadata',type=Path,required=True,help='New separate measurement JSON file')
    p.add_argument('--assets',type=Path,help='Enable one measured invocation using this config')
    p.add_argument('--input',type=Path,help='Release input for a new invocation')
    p.add_argument('--mode',choices=['features','vrs'])
    p.add_argument('--start',type=int)
    p.add_argument('--count',type=int)
    a=p.parse_args()
    if a.metadata.exists():
        p.error('Metadata output already exists')
    wall=None
    if a.assets:
        if a.run.exists() or not a.input or not a.mode:
            p.error('Capture requires --input, --mode and a fresh --run directory')
        argv=[sys.executable,'-B',str(ROOT/'release/cli.py'),'--assets',str(a.assets),
              a.mode,str(a.input),'--output',str(a.run)]
        if a.mode=='vrs':
            if a.start is None or a.count is None:
                p.error('Provide --start and --count for VRS capture')
            argv+=['--start',str(a.start),'--count',str(a.count)]
        start=perf_counter()
        subprocess.run(argv,check=True)
        wall=perf_counter()-start
    elif a.input or a.mode or a.start is not None or a.count is not None:
        p.error('Only --run and --metadata apply when reading existing receipts')
    result=inspect(a.run)
    result.update(external_process_wall_s=wall,hardware=known_hardware() if wall is not None else None,
                  child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if wall is not None else None)
    with a.metadata.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')


if __name__=='__main__':
    main()
