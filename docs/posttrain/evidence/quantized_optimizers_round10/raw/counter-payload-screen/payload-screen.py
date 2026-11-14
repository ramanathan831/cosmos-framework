"""Prospectively ordered, exact-state native rank checkpoint format screen."""
import gc, hashlib, json, os, threading, time
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_provenance
from tools.quantized_optimizers.run_deployment import process_tree_memory
root, output = Path(os.environ['ROOT']), Path(os.environ['RESULTS'])
source=root/'gr00t-fused-native-v10/runs/cosmos3_action/joint_ppo/gr00t-master-adamw/integration/checkpoints/iter_000008'
provenance=checkpoint_provenance(str(source))
target={'format':'','world_size':0,'ranks':{'0':b''}}
began=time.monotonic();dcp.load(target,checkpoint_id=source/'optim');state=checkpoint._deserialize_rank_state(target['ranks']['0'],target['format']);source_seconds=time.monotonic()-began
del target;gc.collect()
def equal(a,b):
 if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for key in a:equal(a[key],b[key])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
rows=[]
for trial,version in enumerate((2,3,3,2,2,3)):
 directory=output/f'trial-{trial}-v{version}'
 samples=[];stop=threading.Event()
 def sample():
  while not stop.is_set():
   samples.append({'time_ns':time.time_ns(),**process_tree_memory(os.getpid())});stop.wait(.25)
 thread=threading.Thread(target=sample);thread.start()
 try:
  start=time.monotonic();payload=(checkpoint._serialize_rank_state if version==2 else checkpoint._serialize_tensor_rank_state)(state);encoded=time.monotonic()
  record={'format':f'cosmos_rank_master_v{version}','world_size':4,'ranks':{'0':payload}}
  metadata=dcp.save(record,checkpoint_id=directory);written=time.monotonic()
  assert type(metadata.state_dict_metadata['ranks.0']).__name__=='BytesStorageMetadata'
  del record,payload;gc.collect()
  read_start=time.monotonic();restored={'format':'','world_size':0,'ranks':{'0':b''}};dcp.load(restored,checkpoint_id=directory);read=time.monotonic()
  decoded=checkpoint._deserialize_rank_state(restored['ranks']['0'],restored['format']);decoded_at=time.monotonic()
  equal(state,decoded);checked=time.monotonic();del restored,decoded;gc.collect()
 finally:
  stop.set();thread.join()
 row={'trial':trial,'format':version,'serialize_seconds':encoded-start,'write_seconds':written-encoded,'read_seconds':read-read_start,'decode_seconds':decoded_at-read,'exact_comparison_seconds':checked-decoded_at,'operation_seconds':written-start+decoded_at-read_start,'complete_seconds':checked-start,'directory_bytes':sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),'peak_pss_bytes':max(s['process_tree_pss'] for s in samples),'peak_rss_bytes':max(s['process_tree_rss'] for s in samples),'same_exact_state':True}
 (output/f'trial-{trial}-memory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples));rows.append(row)
 (output/'completed-trials.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
if checkpoint_provenance(str(source))!=provenance:raise RuntimeError('source checkpoint changed during screen')
(output/'payload-screen.json').write_text(json.dumps({'source_provenance':provenance,'source_read_decode_seconds':source_seconds,'scope':'CPU prepared rank0 native state, 3 prospectively paired format trials; not model throughput or cold delivery','trials':rows},indent=2)+'\n')
