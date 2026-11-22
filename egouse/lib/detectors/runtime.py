"""Offline detector stage measurements; not streaming or device power claims."""
import time,resource
import torch
class DetectorRuntime:
  def __init__(self):
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    self.started=time.perf_counter();self.inference_s=0.;self.frames=0;self.initialization_s=None
  def initialized(self):
    if torch.cuda.is_available():torch.cuda.synchronize()
    self.initialization_s=time.perf_counter()-self.started
  def infer(self,backend,image):
    start=time.perf_counter();result=backend.infer(image)
    if torch.cuda.is_available():torch.cuda.synchronize()
    self.inference_s+=time.perf_counter()-start;self.frames+=1
    return result
  def report(self):
    return dict(initialization_s=self.initialization_s,inference_s=self.inference_s,frames=self.frames,stage_wall_s=time.perf_counter()-self.started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,measurement='offline synchronized inference; initialization separate; includes model memory, not device power')
