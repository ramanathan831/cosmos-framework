import collections,datetime,hashlib,json,pathlib
r=pathlib.Path('/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam')
root=r/'cosmos-optimizer-r6-20261109T1843'
src=r/'cosmos-optimizer-outcomes-20261104T1850/data/train-mixed.jsonl'
ref=r/'cosmos-optimizer-outcomes-20261104T1850/prepare-evaluation-v1/prepared-corpus'
blob=(ref/'manifest.json').read_bytes(); sha=hashlib.sha256(blob).hexdigest()
assert sha=='ea2e64893ab3d7da0f4c8be33001ae99c96dfd2c54fe293449c29255cd1f2d70'
m=json.loads(blob); groups=collections.defaultdict(list)
for row in map(json.loads,src.read_text().splitlines()):
 for msg in row['messages']:
  if isinstance(msg['content'],list):
   for part in msg['content']:
    if part.get('type')=='video': part['video']=str(pathlib.Path(part['video']).resolve())
 t=dict(messages=row['messages'], tools=row.get('tools'), supervise=row.get('supervise','all_assistant'), opener=row.get('opener','self'))
 key=hashlib.sha256(json.dumps(t,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 assert key in m['entries']
 media=[c['video'] for msg in t['messages'] if isinstance(msg['content'],list) for c in msg['content'] if c.get('type')=='video']
 assert len(media)==1
 groups[media[0]].append((key,m['entries'][key]['supervised_tokens'],row))
ranked=sorted((max(t[1] for t in ts),v,ts) for v,ts in groups.items() if len(ts)==2 and min(t[1] for t in ts)<=3)
indices=[round(i*(len(ranked)-1)/7) for i in range(8)]
chosen=[]; selected=[]
for n,i in enumerate(indices):
 length,v,ts=ranked[i]
 for key,tok,row in sorted(ts,key=lambda x:(-x[1],x[0]))[:(1 if n==7 else 2)]:
  selected.append(row);chosen.append(dict(target=key,supervised_tokens=tok,sequence_length=m['entries'][key]['sequence_length'],video=v,video_sha256=m['media'][v],oracle_payload_sha256=m['entries'][key]['sha256']))
p=root/'preparation-slice.jsonl';p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in selected))
plan=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Training/development slice only; metadata selection without task scores. Not a full-corpus identity/reliability qualification.',selection='Eight equispaced order statistics of (substantive token count, video path) among training videos with exactly substantive/choice pair; both targets for first seven, substantive only for eighth. Original target semantics, 15 targets/8 videos.',source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),data=str(p),data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reference=str(ref),reference_sha256=sha,targets=chosen,workers=4,threads_per_worker=2,cache_state='Fresh output and process; source/shared storage cache uncontrolled, no eviction. Profile first; later candidate/control are counterbalanced within the same allocation.',profile_allocation_wall_seconds=420,profile_termination_grace_seconds=120,preparation_total_ceiling_seconds=1200,processor_identity_sha256=hashlib.sha256(json.dumps(m['packer'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
(root/'preparation_slice_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(targets=len(chosen),videos=len(indices),supervised_tokens=[x['supervised_tokens'] for x in chosen],plan_sha256=hashlib.sha256((root/'preparation_slice_plan.json').read_bytes()).hexdigest())))
