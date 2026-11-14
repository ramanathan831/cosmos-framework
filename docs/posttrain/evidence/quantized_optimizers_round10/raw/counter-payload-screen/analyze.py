"""Use the existing paired Student-t interval; no native-delivery extrapolation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tools.quantized_optimizers.analyze_development import interval
root=Path(__file__).resolve().parents[2]
raw=Path(__file__).resolve().parent
screen=json.loads((raw/'payload-screen-v1/payload-screen.json').read_text())
rows=screen['trials'];pairs=((0,1),(3,2),(4,5))
ratios=[rows[a]['complete_seconds']/rows[b]['complete_seconds'] for a,b in pairs]
result={'scope':'One CPU node/session, actual native rank0 state (>4GiB), three counterbalanced format pairs. Includes serialization, write, read, decode, GC gap and exact comparison; excludes initial source read and pinned test startup. Not cold storage, full model delivery or optimizer speed.','source_provenance':screen['source_provenance'],'all_exact':all(x['same_exact_state'] for x in rows),'analysis':'Existing analyze_development.interval: mean paired ratios, two-sided95% Student-t, df2. Three timings are not model seeds or independent nodes.','complete_operation_ratio_v2_over_v3':interval(ratios),'pairs':[{'v2_trial':a,'v3_trial':b,'ratio':ratio} for (a,b),ratio in zip(pairs,ratios)],'mean_phase_seconds':{},'mean_peak_pss_gib':{},'directory_bytes':{},'gate_lower95_above_1p05':interval(ratios)['ci95'][0]>1.05,'limitations':['Same preloaded CPU source state; shared filesystem/cache effects visible','Only rank0 CPU roundtrip measured; four-rank native save/load still requires accelerator validation','No improvement to task metrics is implied']}
for v in (2,3):
 group=[x for x in rows if x['format']==v]
 result['mean_phase_seconds'][str(v)]={k:float(np.mean([x[k] for x in group])) for k in ['serialize_seconds','write_seconds','read_seconds','decode_seconds','exact_comparison_seconds','complete_seconds']}
 result['mean_peak_pss_gib'][str(v)]=float(np.mean([x['peak_pss_bytes']/2**30 for x in group]))
 result['directory_bytes'][str(v)]=sorted(set(x['directory_bytes'] for x in group))
(root/'checkpoint_payload_summary.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(8.3,3.4))
for ax,key,scale,label in zip(axes,['complete_seconds','peak_pss_bytes'],[1,2**30],['CPU roundtrip, seconds','Process peak PSS, GiB']):
 for i,(a,b) in enumerate(pairs):ax.plot([0,1],[rows[a][key]/scale,rows[b][key]/scale],'o-',alpha=.8,label=f'Pair{i+1}')
 ax.set_xticks([0,1],['v2 byte chunks','v3 byte tensor']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.25)
axes[0].legend(frameon=False);fig.suptitle('Native rank0 checkpoint format screen — same exact state; one CPU node')
fig.tight_layout();fig.savefig(root/'checkpoint_payload_screen.png',dpi=180);plt.close(fig)
print(json.dumps({k:result[k] for k in ['complete_operation_ratio_v2_over_v3','mean_phase_seconds','mean_peak_pss_gib','directory_bytes']},indent=2))
