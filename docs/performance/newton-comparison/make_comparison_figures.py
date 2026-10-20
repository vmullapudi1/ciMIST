"""Render three-way PR resource comparisons and retained tree structures."""
from pathlib import Path
import csv
import gzip
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
evidence = ROOT/'retained-tree-comparisons.json'
DATA = json.loads(evidence.read_text() if evidence.exists() else gzip.decompress((ROOT/'retained-tree-comparisons.json.gz').read_bytes()))
PDZ_FINAL = json.loads((ROOT/'pdz-final-export-summary.json').read_text())['measurements']['candidate']['scientific']
COLORS = ['#9199a4', '#197c80', '#b95b30']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype':'none'})
metrics = {
 'pdz': [[1197.897,25215.830,142.745],[483.604,18148.814,26.666],
         [PDZ_FINAL['wall_seconds'],PDZ_FINAL['cpu_seconds'],PDZ_FINAL['peak_process_rss_bytes']/2**30]],
 'mglu5': [[62821.722,388136.236,363.904892],[11741.805,307984.806,57.390],[7055.840,14676.622,38.881504]],
}
if 'eralpha_full' in DATA:
 er=json.loads((ROOT/'er-full-scientific-measurement.json').read_text())
 metrics['eralpha_full']=[[10585.319,223157.764,496.227],[4090.280,166832.646,47.244],
                         [er['wall_seconds'],er['cpu_seconds'],er['peak_process_rss_bytes']/2**30]]

names={'pdz':'PDZ','eralpha_full':'ERα','mglu5':'mGlu5'}
metrics={key:metrics[key] for key in names if key in metrics}
records=[]

def tree_map(ax,d):
 nodes=d['baseline']['nodes'];index={n:i+1 for i,n in enumerate(nodes)}
 b={tuple(sorted(e)) for e in d['baseline']['edges']}
 n={tuple(sorted(e)) for e in d['newton']['edges']}
 for edges,color,label,marker in [(b&n,'#8b95a3','Shared','o'),(b-n,'#286da8','BFGS / equivalent only','o'),(n-b,'#c45b2c','Newton only','x')]:
  xy=[sorted([index[u],index[v]]) for u,v in sorted(edges)]
  if xy:
   x,y=np.array(xy).T
   ax.scatter(x,y,c=color,label=label,s=9 if len(nodes)<600 else 4,marker=marker,alpha=.8,linewidths=.65)
 ax.set(xlabel='Residue index in common ordering',ylabel='Residue index in common ordering',title='Selected tree edges (adjacency map)',xlim=(0,len(nodes)+1),ylim=(0,len(nodes)+1))
 ax.legend(fontsize=7,loc='lower right')

for key,values in metrics.items():
 d=DATA[key];name=names[key]
 labels=['Mainline','Sparse BFGS\nEquivalent PR','Newton\nPR']
 if key=='mglu5':labels=['Mainline\nOOM','Dense BFGS\nBatched','Sparse BFGS\nEquiv. PR','Newton\nPR']
 plot_values = ([[np.nan,np.nan,np.nan]] + values) if key=='mglu5' else values
 plot_colors = (['#dddddd'] + COLORS) if key=='mglu5' else COLORS
 fig,axes=plt.subplots(2,3,figsize=(13.5,8.8))
 for ax,col,title,divisor in zip(axes[0],[0,2,1],['Wall time (minutes)','Peak RAM (GiB)','CPU time (minutes)'],[60,1,60]):
  ys=[row[col]/divisor for row in plot_values]
  bars=ax.bar(labels,ys,color=plot_colors)
  ax.bar_label(bars,labels=[f'{v:,.2f}' if np.isfinite(v) else '' for v in ys],padding=3,fontsize=9)
  ax.set(title=title,ylim=(0,np.nanmax(ys)*1.22))
  ax.tick_params(axis='x',labelsize=8)
  ax.set_xlim(-.6,len(labels)-.4)
  ax.set_xticks(range(len(labels)),labels)
  if key=='mglu5':ax.text(0,np.nanmax(ys)*.1,'Did not\ncomplete',ha='center',fontsize=8,color='#843c18')
 entropy=[d['baseline']['entropy'],d['equivalent']['entropy'],d['newton']['entropy']]
 se=[d['baseline']['entropy_se'],d['equivalent']['entropy_se'],d['newton']['entropy_se']]
 if key=='mglu5':entropy=[np.nan]+entropy;se=[np.nan]+se
 bars=axes[1,0].bar(labels,entropy,color=plot_colors,yerr=se,capsize=3)
 axes[1,0].bar_label(bars,labels=[f'{v:,.3f}' if np.isfinite(v) else '' for v in entropy],padding=5,fontsize=9)
 axes[1,0].set(title='Entropy ± reported SE (nats)',ylim=(0,np.nanmax(np.array(entropy)+se)*1.24))
 axes[1,0].tick_params(axis='x',labelsize=8)
 axes[1,0].set_xlim(-.6,len(labels)-.4)
 axes[1,0].set_xticks(range(len(labels)),labels)
 if key=='mglu5':axes[1,0].text(0,np.nanmax(entropy)*.1,'No result',ha='center',fontsize=8,color='#843c18')
 total=d['residues']-1;shared=d['shared_edges']
 axes[1,1].barh(['Newton vs\nBFGS / equivalent'],[shared/total*100],color=COLORS[1],label=f'Shared: {shared:,}')
 axes[1,1].barh(['Newton vs\nBFGS / equivalent'],[(total-shared)/total*100],left=[shared/total*100],color=COLORS[2],label=f'Replaced: {total-shared:,}')
 axes[1,1].text(shared/total*50,0,f'{shared/total:.1%}',ha='center',va='center',color='white',weight='bold')
 axes[1,1].set(title=f'Tree overlap ({total:,} edges)',xlabel='Fraction of tree edges (%)',xlim=(0,100),ylim=(-.7,.9))
 axes[1,1].legend(loc='upper center',fontsize=9)
 tree_map(axes[1,2],d)
 fig.suptitle(f'{name}: Newton vs equivalent-performance and BFGS\n{d["residues"]:,} residues × {d["frames"]:,} frames; K=100, seed 0',fontsize=15)
 subtitle=('Unmodified mainline ran out of memory; mGlu5 reference is batched dense BFGS (EM16/pair128).'
           if key=='mglu5' else 'BFGS reference: unmodified mainline. Equivalent PR preserves its audited scientific outputs.')
 fig.text(.5,.882,subtitle,ha='center',fontsize=10,color='#843c18',weight='bold')
 foot='Benchmarks: compression disabled (--hdf5_compression none). Default output: lossless gzip-1 + shuffle.\nSingle runs with hardware/thread/allocation differences; whole-configuration observations, not isolated solver speedups.\nTree diagrams show residue-index adjacency, not spatial contacts. Error bars are reported entropy SE, not timing variability.'
 if key=='pdz':foot+='\nPDZ Newton resource measurement is the final block-export composition on A089; BFGS/equivalent ran on different nodes.'
 fig.text(.02,.015,foot,fontsize=9,va='bottom')
 fig.tight_layout(rect=(0,.14,1,.86))
 for ext in ('png','svg','pdf'):
  path=ROOT/f'{key}-newton-comparison.{ext}'
  fig.savefig(path,dpi=180,facecolor='white')
  if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 plt.close(fig)
 reference_label = 'Batched dense BFGS (EM16/pair128); unmodified mainline OOM' if key=='mglu5' else 'Unmodified mainline'
 if key=='mglu5':
  records.append([name,d['residues'],d['frames'],'Unmodified mainline — OOM','','','','none'])
 for label,row in zip([reference_label,'Sparse BFGS / equivalent PR','Newton PR'],values):
  records.append([name,d['residues'],d['frames'],label,*row,'none'])
 with (ROOT/f'{key}-tree-edges.csv').open('w') as f:
  writer=csv.writer(f,lineterminator='\n');writer.writerow(['residue_u','residue_v','membership'])
  b={tuple(sorted(e)) for e in d['baseline']['edges']};n={tuple(sorted(e)) for e in d['newton']['edges']}
  for e in sorted(b|n):writer.writerow([*e,'shared' if e in b&n else ('BFGS_equivalent_only' if e in b else 'Newton_only')])

with (ROOT/'performance.csv').open('w') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['system','residues','frames','configuration','wall_s','cpu_s','peak_rss_GiB','hdf5_compression']);w.writerows(records)

lines=['# Newton: three-system performance and tree comparisons','',
 'Benchmarks used compression-disabled output; current default output is lossless gzip-1/shuffle. All fits use K=100 and seed 0. These single-run whole-configuration comparisons differ in hardware, threading or allocation; they are not controlled solver-only speedups.','',
 '**mGlu5:** unmodified mainline ran out of memory. Its completed batched dense-BFGS reference is reported separately, not substituted into the mainline column.','']
for col,title,unit in [(0,'Wall time','s'),(1,'CPU time','s'),(2,'Peak RAM','GiB')]:
 lines += [f'## {title}', '', f'| System (residues × frames) | Mainline ({unit}) | Sparse BFGS / equivalent PR ({unit}) | Newton PR ({unit}) |', '|---|---:|---:|---:|']
 for key,v in metrics.items():
  d=DATA[key];main='OOM — no completed run' if key=='mglu5' else f'{v[0][col]:,.3f}'
  lines.append(f'| {names[key]} ({d["residues"]:,} × {d["frames"]:,}) | {main} | {v[1][col]:,.3f} | {v[2][col]:,.3f} |')
 if 'eralpha_full' not in metrics:
  lines.append(f'| ERα (500 × 11,082) | {[10585.319,223157.764,496.227][col]:,.3f} | {[4090.280,166832.646,47.244][col]:,.3f} | Running — full trajectory |')
 lines += ['']
lines += ['## Additional runnable mGlu5 reference','',
 '| Configuration | Wall s | CPU s | Peak RAM GiB |','|---|---:|---:|---:|',
 '| Dense BFGS + EM16/pair128 memory batching | 62,821.722 | 388,136.236 | 363.905 |','',
 'This modified reference completed; unmodified mainline did not. Its audited outputs match sparse BFGS / equivalent-performance.','',
 '## Observed Newton improvements','',
 '| System | Wall speedup vs mainline | Wall speedup vs sparse BFGS | Peak RAM reduction vs sparse BFGS | CPU-time reduction vs sparse BFGS |','|---|---:|---:|---:|---:|']
for key,v in metrics.items():
 main_speedup = 'Unavailable — mainline OOM' if key=='mglu5' else f'{v[0][0]/v[2][0]:.2f}×'
 lines.append(f'| {names[key]} | {main_speedup} | {v[1][0]/v[2][0]:.2f}× | {1-v[2][2]/v[1][2]:.2%} | {1-v[2][1]/v[1][1]:.2%} |')
lines += ['','## Scientific results','',
 '| System | Mainline entropy ± SE (nats) | Sparse BFGS entropy ± SE | Newton entropy ± SE | Newton / sparse BFGS shared edges | MI Spearman |',
 '|---|---:|---:|---:|---:|---:|']
for key in metrics:
 d=DATA[key];b=d['baseline'];n=d['newton'];base=f'{b["entropy"]:.6f} ± {b["entropy_se"]:.6f}';main='OOM — no result' if key=='mglu5' else base
 lines.append(f'| {names[key]} | {main} | {base} | {n["entropy"]:.6f} ± {n["entropy_se"]:.6f} | {d["shared_edges"]:,}/{d["residues"]-1:,} ({d["shared_edges"]/(d["residues"]-1):.2%}) | {d["mi_spearman"]:.6f} |')
lines += ['','Mainline and equivalent-performance have identical retained tree edges, entropy, reported SE and MI arrays on PDZ and ERα. For mGlu5 this equality is against the batched dense-BFGS reference. Newton changes these outputs; higher fitted likelihood does not establish biological superiority. Shared-edge counts use undirected residue-name pairs; MI rank correlation uses identical pair ordering.','']
lines += ['## EM solver diagnostics','',
 '| System | Mainline failed / recorded M-steps | Sparse BFGS failed / recorded M-steps | Newton failed / recorded M-steps |',
 '|---|---:|---:|---:|']
diagnostics={'pdz':('426 / 1,261','0 / 1,343'),'mglu5':('9,103 / 20,614','0 / 23,117')}
if 'eralpha_full' in metrics:
 bv=json.loads((ROOT/'er-mainline-validation.json').read_text())
 nv=json.loads((ROOT/'er-full-validation.json').read_text())
 diagnostics['eralpha_full']=(f'{bv["failed_m_steps"]:,} / {bv["recorded_m_steps"]:,}',f'{nv["failed_m_steps"]:,} / {nv["recorded_m_steps"]:,}')
for key in metrics:
 b,n=diagnostics[key]
 if 'recorded_m_steps' in DATA[key]['baseline']:
  base=DATA[key]['baseline'];newton=DATA[key]['newton']
  b=f'{base["failed_m_steps"]:,} / {base["recorded_m_steps"]:,}'
  n=f'{newton["failed_m_steps"]:,} / {newton["recorded_m_steps"]:,}'
 main='OOM — no completed result' if key=='mglu5' else b
 lines.append(f'| {names[key]} | {main} | {b} | {n} |')
lines += ['', 'The solvers have different stopping rules and diagnostics; these are their recorded failures, not a common residual test. The mGlu5 BFGS counts come from the batched dense reference, whose fit is preserved by equivalent-performance. Zero inner failures does not establish a global EM optimum.','']
for key in metrics:lines += [f'## {names[key]} figure', '', f'![{names[key]} comparison]({key}-newton-comparison.png)', '']
lines += ['## Provenance and scope','',
 'The extraction reads retained tree/scalar/MI fields and small EM diagnostic arrays, not the large joint-probability tensors. `retained-tree-comparisons.json` (stored as `.json.gz` on the branch) records artifact paths, sizes, timestamps, selected-field hashes, input manifests and original summaries; CSVs provide the plotted numbers and exact tree-edge membership. This diagnostic is not a new full scientific-equivalence audit.','',
 'PDZ Newton timings use the final block-export candidate from `newton-final-export-pdzfull-20261018` (A089, eight CPUs), rather than the earlier solver-only run. Reference/equivalent timings come from their accepted full-input scientific intervals. mGlu5 uses the final Newton composition and accepted equivalent-performance run.','']
if 'eralpha_full' not in metrics:
 lines += ['**Full ERα Newton benchmark is running as BioHPC job 11182873.** Its full 11,082-frame comparison will be added after completion and validation. The retained 500-frame Newton experiment is supplementary and must not be substituted for the full-trajectory result.','']
else:
 lines += ['Full ERα Newton was newly measured at PR commit `588f0c7`, with all 11,082 frames and compression disabled, using the same input-selection and measurement helpers. `er-full-scientific-measurement.json` records its scientific interval. Input provenance was checked against the accepted equivalent-performance full-ERα run.','']
lines += ['The older 500-frame ERα record is retained in the evidence JSON for provenance but is not plotted or used in these tables.','',
 'Regenerate figures and tables with `MPLCONFIGDIR=/tmp/cimist-mpl python3 make_comparison_figures.py` from this directory (requires Matplotlib and NumPy). The retained snapshots suffice to regenerate the figures; raw trajectories and the complete campaign harness remain in the BioHPC workspace. The launch/extraction scripts document that environment and are not a standalone public benchmark dataset.','']
(ROOT/'README.md').write_text('\n'.join(lines))
intro = (ROOT/'newton-pr-introduction.md').read_text()
tables = '\n'.join(lines).split('## Wall time',1)[1].split('## PDZ figure',1)[0]
interpretation = """
## How the fitted results and trees change

On full PDZ, Newton retains 58 of the 100 BFGS/equivalent-performance tree edges. Its entropy changes from 86.938032 to 87.044021 nats, while reported standard error changes from 1.059569 to 0.855270 nats. The MI rank correlation is 0.930147. A small change in total entropy therefore coexists with substantial changes in selected edges.

On full mGlu5, Newton retains 698 of 1,607 edges from the batched dense-BFGS reference and equivalent-performance fit. Entropy changes from 1394.869247 to 1874.127190 nats and reported standard error from 28.205934 to 39.927457 nats. MI rank correlation is 0.852906. Unmodified mainline did not complete; these are comparisons with the runnable batched reference.

Fresh final-parameter likelihood diagnostics found higher residue likelihoods for all 101 PDZ residues and all 1,608 mGlu5 residues. Gains were 0.454855–14.375457 and 0.614079–11.917854 nats/frame/residue, respectively. Recorded failed inner M-steps changed from 426/1,261 to 0/1,343 on PDZ and from 9,103/20,614 to 0/23,117 on mGlu5. These diagnostics concern the accepted Newton fits preserved by the final composition's equivalence audits. They describe training likelihood, not predictive accuracy or biological superiority.

## Validation and figures

All 11 portable Newton tests passed, covering an independent SciPy Brent reference, bounds, masks, zero-weight handling, objective/gradient checks, ordered forest ties, sparse export, CLI defaults, sparse/dense decoded equality, lossless compression and reader compatibility.

The new comparison reads retained tree edges, entropy, uncertainty, MI arrays and EM diagnostic counts with matched residue ordering. It does not repeat the earlier complete scientific audit or read the large probability tensors.

The [comparison report](../pr-materials/newton-comparison/README.md) contains system-specific figures in PNG, SVG and PDF, plotted values in CSV, and per-edge membership tables. Adjacency diagrams show selected edges by residue index; they are not spatial contact maps.
"""
if 'eralpha_full' not in metrics:
 interpretation += '\n**Full ERα Newton results are pending job 11182873.** The 500-frame historical experiment is not used in these full-trajectory tables.\n'
else:
 d=DATA['eralpha_full'];b=d['baseline'];n=d['newton']
 er_text=f'\nOn full ERα, the new run uses all 11,082 frames at commit `588f0c7`. Newton retains {d["shared_edges"]}/499 BFGS/equivalent tree edges; MI rank correlation is {d["mi_spearman"]:.6f}. Entropy changes from {b["entropy"]:.6f} to {n["entropy"]:.6f} nats, and reported SE from {b["entropy_se"]:.6f} to {n["entropy_se"]:.6f} nats. Input provenance matches the full equivalent-performance run. Execution validation checks shapes, finite outputs, outer convergence and absence of failed Newton M-steps; it is not a claim of BFGS equivalence. Fresh final-parameter likelihood comparisons have not been repeated for this new full ERα run.\n'
 interpretation=interpretation.replace('\nOn full mGlu5,',er_text+'\nOn full mGlu5,')
body = intro+'\n## Wall time'+tables+interpretation
(ROOT/'PR_BODY.md').write_text(body.replace('../pr-materials/newton-comparison/README.md','README.md'))
knowledgebase = ROOT.parent.parent/'knowledgebase'
if knowledgebase.is_dir():
 (knowledgebase/'pr-newton-body.md').write_text(body)
