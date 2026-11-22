"""Offline Medium-33 release command. RGB VRS or exact-contract feature CSV."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

# The script can be invoked by absolute path from any working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release.config import (MANIFEST, load, save, sha, user_paths,
                            verify_child, verify_environment)

START = perf_counter()
F = load(MANIFEST)
HANDS = ('left', 'right')
VOCAB = F['vocabulary']


def verify_weight(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Checkpoint hash differs from release manifest: {path}')


def asset_check(assets, rgb):
    versions=verify_environment('main',F['runtime_versions']['main'])
    verify_weight(assets['student_checkpoint'],F['student']['sha256'])
    if not rgb:
        return {'main':versions}
    verify_weight(assets['hand_checkpoint'],F['hand']['sha256'])
    verify_weight(assets['object_checkpoint'],F['object']['checkpoint_sha256'])
    head=subprocess.check_output(['git','-C',str(assets['hand_repository']),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(assets['hand_repository']),'status','--porcelain','--untracked-files=no'],text=True)
    if head!=F['hand']['external_commit'] or dirty:
        raise ValueError('100DOH source does not match the pinned clean checkout')
    return dict(main=versions,**{r:verify_child(r,assets[f'{r}_python'],F['runtime_versions'][r])
                                for r in ('hand','object')})


def hand_worker(folder,assets):
    import cv2
    import torch
    from lib.detectors.doh100 import Doh100Backend
    from lib.detectors.runtime import DetectorRuntime
    cv2.setNumThreads(1)
    torch.set_num_threads(F['threads'])
    assert torch.cuda.is_available()
    verify_weight(assets['hand_checkpoint'],F['hand']['sha256'])
    runtime=DetectorRuntime()
    model=Doh100Backend(checkpoint=F['hand']['checkpoint_number'],thresh_hand=F['hand']['threshold'],thresh_obj=1.,
                        vendor_root=assets['hand_repository'],weights_path=assets['hand_checkpoint'])
    runtime.initialized()
    frames=load(folder/'clock.json')
    with (folder/'hands.jsonl').open('w') as dst:
        for i,ts in enumerate(frames):
            bgr=cv2.imread(str(folder/f'{i:05d}.jpg'))
            ds=runtime.infer(model,bgr)
            hands=[dict(box=list(d.box),score=d.score,side=d.side,contact=d.contact,offset=list(d.offset),
                        raw_fields=list(d.raw_fields)) for d in ds if d.label=='hand']
            dst.write(json.dumps(dict(timestamp_ns=ts,hands=hands))+'\n')
    save(folder/'hands-runtime.json',dict(runtime.report(),python_start_to_exit_s=perf_counter()-START,
                                         torch=torch.__version__))


def object_worker(folder,out,assets):
    import cv2
    import torch
    from lib.detectors.closed_set import ClosedSetYoloBackend,detection_record,validate_object_records
    from lib.detectors.contracts import parse_doh_hand_row
    from lib.detectors.runtime import DetectorRuntime
    assert torch.cuda.is_available()
    cv2.setNumThreads(1)
    torch.set_num_threads(F['threads'])
    with (folder/'hands.jsonl').open() as f:
        rows=[json.loads(line) for line in f]
    runtime=DetectorRuntime()
    model=ClosedSetYoloBackend(dict(F['object'],checkpoint=str(assets['object_checkpoint']),vocabulary=VOCAB))
    runtime.initialized()
    with (out/'detections.jsonl').open('w') as dst:
        for i,(row,ts) in enumerate(zip(rows,load(folder/'clock.json'),strict=True)):
            assert row['timestamp_ns']==ts
            for raw in row['hands']:
                hand=parse_doh_hand_row(raw['raw_fields'])
                assert hand.side==raw['side'] and list(hand.offset)==raw['offset']
            row['objects']=[detection_record(d) for d in runtime.infer(model,cv2.imread(str(folder/f'{i:05d}.jpg')))]
            validate_object_records(row['objects'],VOCAB)
            dst.write(json.dumps(row,separators=(',',':'))+'\n')
    save(out/'objects-runtime.json',dict(runtime.report(),python_start_to_exit_s=perf_counter()-START,
                                         torch=torch.__version__))


def load_numeric(features):
    import numpy as np
    import pandas as pd
    from lib.student_sequence_data import hand_stream_from_features_csv
    from lib.student_features import FEATURE_COLUMNS
    table=pd.read_csv(features)
    t=table[['timestamp_ns','t_sec','hand_side','object_name']]
    assert not t.duplicated(['timestamp_ns','hand_side','object_name']).any()
    assert set(t['object_name'])==set(VOCAB) and set(t['hand_side'])==set(HANDS)
    assert t['timestamp_ns'].dtype.kind in 'iu'
    assert len(t)==t['timestamp_ns'].nunique()*2*len(VOCAB)
    assert table['y_interacting'].isna().all()
    assert not np.isinf(table[FEATURE_COLUMNS].to_numpy()).any()
    cache=[]
    for side in HANDS:
        x,_,seconds,ts=hand_stream_from_features_csv(features,side,VOCAB,
                  {v:i+1 for i,v in enumerate(VOCAB)},allow_unknown_labels=True)
        assert np.allclose(seconds,(np.array(ts)-ts[0])/1e9,atol=1e-10,rtol=0)
        cache.append((np.ascontiguousarray(x.transpose(1,0,2)),np.asarray(ts,dtype=np.int64)))
    assert np.array_equal(cache[0][1],cache[1][1])
    assert (np.diff(cache[0][1])>0).all()
    return [it[0] for it in cache],cache[0][1]


def model_load(checkpoint):
    import torch
    from lib.pairwise_gru import PairwiseGRU
    device='cpu'
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    assert ck['vocabulary']==VOCAB and ck['config_sha256']==F['training_config_sha256']
    model=PairwiseGRU()
    model.load_state_dict(ck['model_state'],strict=True)
    return model.eval()


def forward(model,x,chunk=None):
    import torch
    import numpy as np
    state=None
    out=[]
    width=chunk or x.shape[1]
    with torch.inference_mode():
        for start in range(0,x.shape[1],width):
            it=torch.from_numpy(x[:,start:start+width].copy())
            values,state=model(it,torch.full((it.shape[0],),it.shape[1],dtype=torch.long),state)
            out.append(values)
    return torch.cat(out,dim=1).T.contiguous()


def decode(prob,clock,chunk=None):
    import numpy as np
    flags=np.empty(prob.shape,bool)
    last=None
    active=np.zeros(prob.shape[::2],bool)
    v=prob.astype(np.float64,copy=False)
    width=chunk or len(clock)
    for k in range(0,len(clock),width):
        for i in range(k,min(k+width,len(clock))):
            t=clock[i]
            if last is None or t-last>F['decoder']['clock_gap_ns']:
                active.fill(False)
            active=np.where(active,v[:,i]>=F['decoder']['release'],v[:,i]>=F['decoder']['onset'])
            flags[:,i]=active
            last=t
    return flags


def intervals(mask,clock,scope):
    import numpy as np
    period=int(np.median(np.diff(clock))) if len(clock)>1 else 33_333_333
    edges=np.r_[clock,clock[-1]+period]
    gaps=np.flatnonzero(np.diff(clock)>F['decoder']['clock_gap_ns'])+1
    result=[]
    for j,name in enumerate(VOCAB):
        for a,b in np.flatnonzero(np.diff(np.r_[False,mask[:,j],False])).reshape(-1,2):
            cuts=[a,*gaps[(a<gaps)&(gaps<b)],b]
            for start,end in zip(cuts,cuts[1:]):
                stop=min(int(edges[end]),int(edges[end-1])+period) if end in gaps else int(edges[end])
                stamp=int(clock[start])
                result.append(dict(hand=scope,object=name,class_id=j,start_frame=int(start),end_frame_exclusive=int(end),
                       start_ns=stamp,end_ns=stop,start_s=(stamp-int(clock[0]))/1e9,end_s=(stop-int(clock[0]))/1e9))
    return result


def cached(features,out,checkpoint):
    import numpy as np
    import torch
    torch.set_num_threads(F['threads'])
    t=perf_counter()
    m=model_load(checkpoint);loaded=perf_counter()
    inputs,clock=load_numeric(features)
    built=perf_counter()
    z=torch.stack([forward(m,x) for x in inputs]);prob=torch.sigmoid(z).numpy()
    model_done=perf_counter()
    decisions=decode(prob,clock)
    merged=decisions.any(axis=0)
    np.savez_compressed(out/'predictions.npz',clock=clock,logits=z.numpy(),probability=prob,on=decisions,merged=merged,
                        hands=np.asarray(HANDS),vocabulary=np.asarray(VOCAB))
    seq=[*intervals(decisions[0],clock,'left'),*intervals(decisions[1],clock,'right'),*intervals(merged,clock,'merged')]
    save(out/'intervals.json',dict(origin_ns=int(clock[0]),semantics='half-open',intervals=seq))
    timing=dict(device='CPU float32',frames=len(clock),load_model_and_verify_s=loaded-t,feature_read_tensorize_s=built-loaded,
                model_forward_and_sigmoid_s=model_done-built,decode_and_export_s=perf_counter()-model_done,
                cached_pipeline_s=perf_counter()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                torch=torch.__version__)
    save(out/'student-runtime.json',timing)


def run_vrs(source,start,count,out,asset_path,assets):
    import cv2
    import numpy as np
    import torch
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain,TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId
    from lib.pair_features import build_features_deploy_grid
    from lib.student_features import FEATURE_COLUMNS,FEATURE_SCHEMA
    assert FEATURE_SCHEMA==F['feature_schema']
    cv2.setNumThreads(1)
    torch.set_num_threads(F['threads'])
    assert torch.cuda.is_available()
    t=perf_counter()
    with TemporaryDirectory(dir=out) as scratch:
        root=Path(scratch)
        reader=data_provider.create_vrs_data_provider(str(source))
        rgb=StreamId('214-1')
        timestamps=list(reader.get_timestamps_ns(rgb,TimeDomain.TIME_CODE))[start:start+count]
        assert len(timestamps)==count and (np.diff(timestamps)>0).all()
        save(root/'clock.json',timestamps)
        for i,ts in enumerate(timestamps):
            img,record=reader.get_image_data_by_time_ns(rgb,ts,TimeDomain.TIME_CODE,TimeQueryOptions.CLOSEST)
            bgr=cv2.cvtColor(cv2.rotate(img.to_numpy_array(),cv2.ROTATE_90_CLOCKWISE),cv2.COLOR_RGB2BGR)
            assert bgr.shape==(1408,1408,3)
            assert cv2.imwrite(str(root/f'{i:05d}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,F['jpeg_quality']])
        decode_done=perf_counter()
        torch_lib=assets['hand_python'].parent.parent/'lib/python3.10/site-packages/torch/lib'
        env=dict(os.environ,LD_LIBRARY_PATH=str(torch_lib)+
                 (':'+os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else ''))
        prefix=['-B',str(Path(__file__).resolve()),'--assets',str(asset_path)]
        subprocess.run([str(assets['hand_python']),*prefix,'_hand',str(root)],env=env,check=True)
        hand_done=perf_counter()
        save(out/'hands-runtime.json',load(root/'hands-runtime.json'))
        # Use the same separate vendor runtime as historical object extraction.
        (out/'hands.jsonl').write_bytes((root/'hands.jsonl').read_bytes())
        subprocess.run([str(assets['object_python']),*prefix,'_object',str(root),str(out)],check=True)
        with (out/'detections.jsonl').open() as f:
            records=[json.loads(line) for line in f]
        object_done=perf_counter()
    values=build_features_deploy_grid({r['timestamp_ns']:r for r in records},VOCAB,origin_ns=timestamps[0])
    assert len(values)==count*len(VOCAB)*2
    values[FEATURE_COLUMNS]=values[FEATURE_COLUMNS].astype('float32')
    path=out/'features.csv.gz'
    values.to_csv(path,index=False,compression={'method':'gzip','compresslevel':5,'mtime':0})
    feat_done=perf_counter()
    cached(path,out,assets['student_checkpoint'])
    save(out/'pipeline-runtime.json',dict(frames=count,resolution=[1408,1408],threads=F['threads'],source=str(source),
            source_start_frame=start,decode_and_jpeg_s=decode_done-t,hand_process_wall_s=hand_done-decode_done,
            object_stage_wall_s=object_done-hand_done,features_s=feat_done-object_done,
            end_to_end_s=perf_counter()-START,opencv=cv2.__version__,environment=dict(torch=torch.__version__)))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--assets',required=True,type=Path,help='Explicit local asset path JSON')
    sub=cli.add_subparsers(dest='command',required=True)
    for command, description in [('features','Offline CPU inference from exact deployment features'),
                                 ('vrs','Offline GPU detection and CPU Student inference from Aria RGB VRS')]:
        p=sub.add_parser(command,help=description)
        p.add_argument('input',type=Path)
        p.add_argument('--output',type=Path,required=True)
        if command=='vrs':
            p.add_argument('--start',type=int,required=True)
            p.add_argument('--count',type=int,required=True)
    # Private process dispatch, only used by the two serial vendor environments.
    if len(sys.argv)>=5 and sys.argv[1]=='--assets' and sys.argv[3] in ('_hand','_object'):
        paths=user_paths(Path(sys.argv[2]).resolve())
        if sys.argv[3]=='_hand':
            hand_worker(Path(sys.argv[4]),paths)
        else:
            object_worker(Path(sys.argv[4]),Path(sys.argv[5]),paths)
        return
    a=cli.parse_args()
    asset_path=a.assets.resolve()
    assets=user_paths(asset_path)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]=str(F['threads'])
    versions=asset_check(assets,rgb=a.command=='vrs')
    a.output.mkdir()
    if a.command=='features':
        cached(a.input,a.output,assets['student_checkpoint'])
    else:
        if a.start<0 or a.count<2:
            cli.error('VRS start must be non-negative and count at least two')
        run_vrs(a.input,a.start,a.count,a.output,asset_path,assets)
    save(a.output/'provenance.json',dict(release_sha256=sha(MANIFEST),input=str(a.input.resolve()),
           assets_sha256=sha(asset_path),entry_sha256=sha(__file__),environments=versions,
           command=a.command))
    print(f'Wrote {a.command} outputs to {a.output}')


if __name__=='__main__':
    main()
