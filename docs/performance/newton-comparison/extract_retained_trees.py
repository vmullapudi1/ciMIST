"""Read compact tree/scalar fields only; never load joint probability tensors or refit."""
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/work/greencenter/s191084/cimist_perf/benchmarks/results')
CASES = {
 'pdz': ('pdz3_full', 'newton-final-export-pdzfull-20261018/candidate', 101, 10500),
 'eralpha': ('er_dimer_500', 'newton_jax__er_dimer_500', 500, 500),
 'mglu5': ('memory_combined_em16_pair128__mglu5_full', 'optimizedfinal20261018_newton_candidate_emfull_pair128__mglu5_full', 1608, 3236),
}

EQUIVALENT = {
 'pdz': 'configurable20261018_mainline_candidate_em8_pair128__pdz3_full',
 'mglu5': 'configurable20261018_mainline_candidate_em8_pair128__mglu5_full',
 'eralpha_full': 'configurable20261018_mainline_candidate_em8_pair128__er_dimer_full',
}
new_er = 'prnewton588f0c7_review20261020__er_dimer_full'
if (ROOT/new_er/'SUCCESS.json').exists():
 CASES['eralpha_full'] = ('er_dimer_full', new_er, 500, 11082)

def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def read_run(name):
 root=ROOT/name
 paths=list(root.glob('instrumented/outputs/RESULTS_ciMIST.h5'))
 if not paths:
  paths=list(root.glob('**/RESULTS_ciMIST.h5'))
 if len(paths)!=1:
  raise RuntimeError((name, [str(p) for p in paths]))
 path=paths[0]
 with h5py.File(path,'r') as f:
  fields={}
  for key in ['entropy','entropy_se','counts_variance','structural_variance']:
   fields[key]=float(f['tree/'+key][()].reshape(-1)[0])
  nodes=f['tree/vertices/name'].asstr()[()].tolist()
  edges=f['tree/edges/name'].asstr()[()].tolist()
  # These one-dimensional MI vectors are small; P tensors remain unread.
  mi=f['MI_graph/edges/I_mle'][()]
  pairs=f['MI_graph/edges/name'].asstr()[()]
  fields.update(nodes=nodes,edges=edges,N_obs=int(f['tree/N_obs'][()].reshape(-1)[0]),
                n_states=f['tree/vertices/N_states'][()].tolist())
  iterations=np.asarray(f['VMM/n_iter'],dtype=int)
  successes=np.asarray(f['VMM/successes'])
  fields['recorded_m_steps']=int(iterations.sum())
  fields['failed_m_steps']=int(sum(np.count_nonzero(successes[i,:count]==0) for i,count in enumerate(iterations)))
  fields['selected_field_hashes']={key:hashlib.sha256(f[key][()].tobytes()).hexdigest()
        for key in ['tree/entropy','tree/entropy_se','MI_graph/edges/I_mle']}
 stat=path.stat()
 fields['artifact']={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
 fields['metadata_hashes']={str(p.relative_to(root)):digest(p) for p in
      [root/'SUCCESS.json',root/'input.json',root/'export_compression.json'] if p.exists()}
 if (root/'input.json').exists():
  fields['input_manifest']=json.loads((root/'input.json').read_text())
 return fields,mi,pairs

out={}
for case,(b,c,n,t) in CASES.items():
 base,mi_b,pairs_b=read_run(b);cand,mi_c,pairs_c=read_run(c)
 assert base['nodes']==cand['nodes'],case
 assert len(base['nodes'])==n and base['N_obs']==cand['N_obs']==t,case
 assert np.array_equal(pairs_b,pairs_c),case
 eb={tuple(sorted(edge)) for edge in base['edges']};ec={tuple(sorted(edge)) for edge in cand['edges']}
 assert len(eb)==len(ec)==n-1,case
 record={'residues':n,'frames':t,'baseline_run':b,'newton_run':c,'baseline':base,'newton':cand,
         'shared_edges':len(eb&ec),'baseline_only_edges':sorted(eb-ec),'newton_only_edges':sorted(ec-eb),
         'edge_jaccard':len(eb&ec)/len(eb|ec),'mi_spearman':float(spearmanr(mi_b,mi_c).statistic),
         'scope':'Read-only retained HDF5 tree/scalar/MI diagnostic. No fit rerun, no full scientific-equivalence audit.'}
 if case in EQUIVALENT:
  equiv,mi_e,pairs_e=read_run(EQUIVALENT[case])
  assert equiv['nodes']==base['nodes'] and equiv['N_obs']==base['N_obs'],case
  assert {tuple(sorted(e)) for e in equiv['edges']}==eb,case
  assert equiv['entropy']==base['entropy'] and equiv['entropy_se']==base['entropy_se'],case
  assert np.array_equal(mi_e,mi_b) and np.array_equal(pairs_e,pairs_b),case
  assert equiv['recorded_m_steps']==base['recorded_m_steps'] and equiv['failed_m_steps']==base['failed_m_steps'],case
  record['equivalent']=equiv
  record['equivalent_run']=EQUIVALENT[case]
  record['equivalent_matches_baseline_in_selected_fields']=True
 summary=ROOT/c/'scientific-comparison-summary.json'
 if summary.exists(): record['historical_solver_comparison']=json.loads(summary.read_text())
 out[case]=record
 print(case, 'shared',record['shared_edges'],'/',n-1,'entropy',base['entropy'],cand['entropy'],'MI rank',record['mi_spearman'],flush=True)
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
