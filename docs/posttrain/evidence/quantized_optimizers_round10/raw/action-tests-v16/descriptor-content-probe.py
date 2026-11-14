import json,os,sys,time,resource
from pathlib import Path
import torch
import torch.distributed as dist

def worker(rank,checkpoint,output):
 from cosmos_framework.posttrain.checkpoint import CheckpointManager
 dist.init_process_group('gloo',init_method='file://'+str(Path(output)/'probe-rendezvous'),rank=rank,world_size=4)
 try:
  begin=time.monotonic();identity=CheckpointManager().publication_identity(checkpoint);elapsed=time.monotonic()-begin
  identities=[None]*4;dist.all_gather_object(identities,identity)
  if any(x!=identity for x in identities):raise RuntimeError('full content identity differs')
  payload={'rank':rank,'checkpoint':checkpoint,'seconds':elapsed,'identity':identity,'all_four_readers_equal':True,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_commit':'677e7e039012b40033e4ea137587619808d12a01','scope':'One descriptor-bound, double-content publication verification per CPU rank of actual completed native checkpoint; no publication mutation'}
  (Path(output)/f'content-probe-rank{rank}.json').write_text(json.dumps(payload,indent=2)+'\n')
 finally:dist.destroy_process_group()
if __name__=='__main__':
 torch.multiprocessing.spawn(worker,args=(sys.argv[1],sys.argv[2]),nprocs=4,join=True)
