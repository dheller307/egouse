"""One causal binary GRU shared across independent hand-object streams."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lib.label_coverage import known_mask


class PairwiseGRU(nn.Module):
    def __init__(self, input_dim=18, hidden=128, dropout=.1):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,16),nn.ReLU())
        self.gru=nn.GRU(16,hidden,2,batch_first=True,dropout=dropout)
        self.head=nn.Linear(hidden,1)

    def forward(self,x,lengths,state=None):
        packed=pack_padded_sequence(self.encoder(x),lengths.cpu(),batch_first=True,enforce_sorted=False)
        output,state=self.gru(packed,state)
        output,_=pad_packed_sequence(output,batch_first=True,total_length=x.shape[1])
        return self.head(output).squeeze(-1),state


def binary_targets(reference,timestamps,hand,vocabulary):
    ref=reference[reference.hand_side==hand]
    if ref.duplicated(['timestamp_ns','object_name']).any():
        raise ValueError('Duplicate pair labels')
    target=np.zeros((len(timestamps),len(vocabulary)),dtype=np.float32)
    mask=np.zeros_like(target,dtype=bool)
    time_ids={int(t):i for i,t in enumerate(timestamps)};objects={o:i for i,o in enumerate(vocabulary)}
    known=known_mask(ref);unsupported=0
    for row,is_known in zip(ref.itertuples(),known):
        if not is_known or row.timestamp_ns not in time_ids:continue
        if row.object_name not in objects:unsupported+=1;continue
        i,j=time_ids[row.timestamp_ns],objects[row.object_name]
        target[i,j]=row.v2_interacting;mask[i,j]=True
    return target,mask,dict(positive=int(target[mask].sum()),negative=int(mask.sum()-target[mask].sum()),unknown=int((~mask).sum()),unsupported_known_pairs=unsupported)


def masked_binary_loss(logits,target,known):
    count=int(known.sum().item())
    if not count:return logits.sum()*0,logits.sum()*0,0
    total=nn.functional.binary_cross_entropy_with_logits(logits[known],target[known],reduction='sum')
    return total/count,total,count
