"""Read-only measured tensor accounting of a retained native fused checkpoint."""
import gc
import json
import os
import time
import resource
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from cosmos_framework.posttrain.checkpoint.master_optimizer import _deserialize_rank_state
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_provenance
root=Path(os.environ['ROOT'])
p=root/'gr00t-fused-native-v10/runs/cosmos3_action/joint_ppo/gr00t-master-adamw/integration/checkpoints/iter_000008'
before=checkpoint_provenance(str(p));start=time.monotonic();rows=[]
for rank in range(4):
 target={'format':'','world_size':0,'ranks':{str(rank):b''}}
 dcp.load(target,checkpoint_id=p/'optim')
 saved=_deserialize_rank_state(target['ranks'][str(rank)],target['format'])
 sizes={};counts={};dtypes={}
 for state in saved['backend']['state'].values():
  for k,v in state.items():
   if isinstance(v,torch.Tensor):
    sizes[k]=sizes.get(k,0)+v.numel()*v.element_size()
    counts[k]=counts.get(k,0)+1
    dtypes.setdefault(k,set()).add(str(v.dtype))
 rows.append({'rank':rank,'world_size':target['world_size'],'bytes_by_state_key':sizes,'tensors_by_state_key':counts,'dtypes_by_state_key':{k:sorted(v) for k,v in dtypes.items()}})
 del saved,target
 gc.collect()
after=checkpoint_provenance(str(p))
if before!=after:raise RuntimeError('checkpoint changed during read-only accounting')
(Path(os.environ['RESULTS'])/'counter-accounting.json').write_text(json.dumps({'scope':'Actual native checkpoint tensors; read-only CPU deserialization, no optimizer step or resave','source':before,'ranks':rows,'seconds':time.monotonic()-start,'process_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2)+'\n')
print(json.dumps(rows))
