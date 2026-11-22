"""Render saved release decisions as a timeline or sync them to source Aria RGB."""

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(intervals, names, duration):
    sides = ('left', 'right', 'merged')
    lanes = [(name,side) for name in names for side in sides]
    fig, ax = plt.subplots(figsize=(8,3.6),dpi=100,layout='constrained')
    for i, (name,side) in enumerate(lanes):
        y = len(lanes)-1-i
        ax.barh(y,duration,color='#e4ebed',height=.62)
        for event in intervals:
            if event['hand']==side and event['object']==name:
                ax.barh(y,event['end_s']-event['start_s'],left=event['start_s'],
                        color='#247e6d',height=.62)
    ax.set(yticks=range(len(lanes)),yticklabels=[f'{name} · {side}' for name,side in reversed(lanes)],
           xlabel='Seconds from first source frame',xlim=(0,duration),
           title='EgoUse decoded interactions')
    ax.spines[['top','right']].set_visible(False)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    x0,x1 = ax.transAxes.transform((0,0))[0],ax.transAxes.transform((1,0))[0]
    bottom,top = [ax.transAxes.transform((0,a))[1] for a in (0,1)]
    return fig,panel,(x0,x1,panel.shape[0]-top,panel.shape[0]-bottom)


def video(source, clock, duration, panel, bounds, output):
    import cv2
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain,TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId
    source = data_provider.create_vrs_data_provider(str(source))
    stream = StreamId('214-1')
    original = set(source.get_timestamps_ns(stream,TimeDomain.TIME_CODE))
    if not set(clock.tolist()) <= original:
        raise ValueError('Prediction clock does not identify frames in the supplied VRS')
    dt = np.diff(clock)
    period = float(np.median(dt))
    if not np.all(np.abs(dt-period) <= .1*period):
        raise ValueError('Video rendering requires near-regular source clocks; use the PNG timeline')
    cv2.setNumThreads(1)
    height,width = panel.shape[:2]
    writer = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                               '-s',f'{width+height}x{height}','-r',str(1e9/period),'-i','-',
                               '-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',
                               '-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
    try:
        for t in clock:
            image,_ = source.get_image_data_by_time_ns(stream,int(t),TimeDomain.TIME_CODE,TimeQueryOptions.CLOSEST)
            rgb = cv2.resize(cv2.rotate(image.to_numpy_array(),cv2.ROTATE_90_CLOCKWISE),(height,height))
            right = panel.copy()
            fraction = (int(t)-int(clock[0]))/1e9/duration
            x = round(bounds[0]+fraction*(bounds[1]-bounds[0]))
            cv2.line(right,(x,round(bounds[2])),(x,round(bounds[3])),(28,42,50),2)
            writer.stdin.write(np.hstack((rgb,right)).tobytes())
    finally:
        writer.stdin.close()
        code=writer.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited {code}; inspect the output before reuse')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--intervals',type=Path,required=True)
    p.add_argument('--object',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-vrs',type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output path')
    if a.output.suffix not in ('.png','.mp4') or (a.output.suffix=='.mp4') != bool(a.source_vrs):
        p.error('Choose a PNG timeline or provide a source VRS and MP4 output')
    with np.load(a.predictions,allow_pickle=False) as z:
        clock, vocab = z['clock'], z['vocabulary'].tolist()
    labels = json.loads(a.intervals.read_text())
    if (not set(a.object)<=set(vocab) or len(a.object)!=len(set(a.object)) or len(clock)<2 or
        not (np.diff(clock)>0).all() or labels['origin_ns']!=int(clock[0]) or labels['semantics']!='half-open'):
        p.error('Object names or interval clock disagree with the saved prediction archive')
    duration = (int(clock[-1])-int(clock[0])+int(np.median(np.diff(clock))))/1e9
    if not all(0<=e['start_s']<e['end_s']<=duration for e in labels['intervals']):
        p.error('Saved intervals extend outside this prediction archive')
    fig,panel,bounds = plot(labels['intervals'],a.object,duration)
    if a.source_vrs:
        video(a.source_vrs,clock,duration,panel,bounds,a.output)
    else:
        fig.savefig(a.output)
    plt.close(fig)


if __name__=='__main__':
    main()
