"""Focused exact regression for scalar jackknife; reference frozen before change."""
import copy
import unittest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from cimist import mist
from cimist.ci.mpnn import ResidueStates
from cimist.sparse_probability import SparseProbability

MIST = mist.MIST
_PresortedDeletionMST = mist._PresortedDeletionMST
CANDIDATE = mist.residue_pseudojackknife

def reference_pseudojackknife(tree):
    prepared = _PresortedDeletionMST(tree.MI_graph, weight="I")
    entropies = []
    contributions = []
    for n in tree.nodes():
        other_nodes = [n_ for n_ in tree.MI_graph.nodes() if n_ != n]
        G = nx.subgraph(tree.MI_graph, other_nodes)

        # estimator="mle" ensures time is not spent re-computing posterior mean entropies
        # with little effect on estimated variance
        prefitted_tree, _ = prepared.tree_without(n)
        tree_resamp = MIST(G, N_obs=tree.N_obs, tree=prefitted_tree, entropy_estimator="mle", calculate_marginals=False, uncertainty=False, prior=tree.prior)

        entropies.append(tree_resamp.entropy())
        contributions.append(tree_resamp.residue_entropies() - 0.5*tree_resamp.residue_sum_MIs())
    contributions_df = pd.concat(contributions, axis=1).map(float)
    contribution_se = contributions_df.std(axis=1, skipna=True)
    return jnp.array(entropies), contributions_df, contribution_se

BASELINE = reference_pseudojackknife

def equal(left, right):
    if isinstance(left, SparseProbability): left = left.dense_numpy()
    if isinstance(right, SparseProbability): right = right.dense_numpy()
    if isinstance(right, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    elif isinstance(right, pd.Series):
        pd.testing.assert_series_equal(left, right, check_exact=True)
    else:
        left, right = np.asarray(left), np.asarray(right)
        assert left.dtype == right.dtype, (left.dtype,right.dtype)
        np.testing.assert_equal(left,right)

def compare(left, right):
    assert set(vars(left)) == set(vars(right))
    for name in vars(right):
        if name not in ('MI_graph','T','nodes','edges'):
            equal(getattr(left,name),getattr(right,name))
    for name in ('MI_graph','T'):
        first, second = getattr(left,name),getattr(right,name)
        assert list(first) == list(second)
        assert list(first.edges) == list(second.edges)
        assert [list(first[n]) for n in first] == [list(second[n]) for n in second]
        assert first.graph == second.graph
        for category in ('nodes','edges'):
            a,b = getattr(first,category),getattr(second,category)
            for key in b:
                assert set(a[key]) == set(b[key])
                for attr in b[key]:equal(a[key][attr],b[key][attr])
    # Existing shared probability/node-marginal relationships must remain.
    def aliases(tree):
        objects=[]
        for graph in (tree.MI_graph,tree.T):
            objects.extend(d['p'] for _,d in graph.nodes(data=True))
            objects.extend(d['P'] for _,_,d in graph.edges(data=True))
        ids={}
        return [ids.setdefault(id(x),len(ids)) for x in objects]
    assert aliases(left)==aliases(right)

def build(factory, variant):
    previous = mist.residue_pseudojackknife
    mist.residue_pseudojackknife = BASELINE if variant=='baseline' else CANDIDATE
    try:return factory()
    finally:mist.residue_pseudojackknife=previous

def checks():
    rng=np.random.default_rng(7620)
    records=[]
    for count in (2,6):
        trajectory=rng.integers(0,4,(count,73),dtype=np.int64)
        trajectory[1]=trajectory[0]  # Exact tied dependence.
        counts=np.stack([np.bincount(row,minlength=4) for row in trajectory])
        states=ResidueStates(jnp.asarray(np.sum(counts>0,axis=1)),jnp.asarray(trajectory),jnp.asarray(counts))
        for prior in ('haldane','percs','jeffreys','laplace'):
            for storage in ('dense','sparse'):
                factory=lambda:mist.MIST.from_residue_states(states,[f'r{i}' for i in range(count)],
                            prior=prior,uncertainty=True,probability_storage=storage,pair_batch_size=4)
                first=build(factory,'baseline');second=build(factory,'candidate');compare(first,second)
                records.append(dict(case='states',residues=count,prior=prior,storage=storage,all_fields_and_aliases_exact=True))
    for kind in ('disconnected','star','path','tied'):
        graph=nx.Graph()
        graph.add_nodes_from((n,dict(S=float(n+.25),p=jnp.asarray([.5,.5]),N_states=2)) for n in range(6))
        pairs=([(0,1),(1,2),(0,2),(3,4)] if kind=='disconnected' else
               [(0,n) for n in range(1,6)] if kind=='star' else
               [(n,n+1) for n in range(5)] if kind=='path' else
               [(u,v) for u in range(6) for v in range(u+1,6)])
        rng.shuffle(pairs)
        for index,(u,v) in enumerate(pairs):
            # Deliberately unequal marginals/orientations expose skipped writes.
            p=np.asarray([[.1,.2],[.3,.4]])
            if index%2:p=p.T
            graph.add_edge(u,v,I=1.,P=SparseProbability.encode(p),axes=(v,u))
        for prior in ('haldane','percs','jeffreys','laplace'):
            factory=lambda:mist.MIST(copy.deepcopy(graph),N_obs=100,uncertainty=True,prior=prior)
            first=build(factory,'baseline');second=build(factory,'candidate');compare(first,second)
            records.append(dict(case=kind,residues=6,prior=prior,storage='sparse',all_fields_and_aliases_exact=True))
    return records


class ScalarJackknifeContract(unittest.TestCase):
    def test_complete_fields_and_aliases_match_frozen_constructor_path(self):
        self.assertEqual(len(checks()), 32)



if __name__ == '__main__':
    unittest.main()
