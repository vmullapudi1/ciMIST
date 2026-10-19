# Safeguarded Newton: improved fits, different outputs

This branch is stacked on `pr/equivalent-performance`. The source hash manifest
now describes the final Newton composition; the inherited equivalence report
and its receipts describe the parent branch, not Newton versus original BFGS.

On full mGlu5 (1,608 residues, 3,236 unchanged frames, seed 0, K=100), the final
composition used 7,055.840 wall seconds, 14,676.622 CPU seconds and 38.8815 GiB.
The batched original BFGS reference used 62,821.722 seconds, 388,136.236 CPU
seconds and 363.9049 GiB: observed 8.90× wall speedup and 89.32% less RAM.
Nodes, thread controls and EM batching differ: this is a whole-configuration
comparison, not a controlled solver-only speedup.

Fresh saved-parameter likelihoods improve in all 1,608 residue models by
0.614079–11.917854 nats per frame per residue relative to the BFGS reference.
Newton has 0 failures in 23,117 M-steps; BFGS has 9,103 failures in 20,614.
This is better observed likelihood, not proof of a global optimum, better
biological estimates, or equivalent states/trees/entropy. Final Newton agrees
exactly with its accepted Newton parent on complete PDZ and mGlu5 scientific
comparisons, including fresh likelihoods and both decoded scientific pickles,
under the unchanged rtol=1e-8, atol=1e-10 gates.

## Model and solver contract

The product von Mises model and bounded conditional M-step are unchanged.
After responsibilities and means are fixed, solve I1(kappa)/I0(kappa)=C/W
independently per active component/angle, using scaled Bessel functions.
Check bounds 10 and 600 first; use a Newton step only inside the bracket with
positive derivative, otherwise bisect. Masked dimensions remain zero;
zero-weight components retain clipped concentrations. Invalid input or failed
iteration retains the prior concentrations. Resultant overshoot within 64 eps
is clipped and reported as status 3; larger overshoot is rejected.

Iterations stop at residual <= max(3e-15,8 eps) or bracket width <=
max(1e-10,8 eps abs(kappa)), with cap min(maxiter,100). BFGS gtol and line-search
limits do not control Newton. Diagnostic counters therefore differ. The
explicit BFGS fallback uses the original frame-based objective, not the parent
branch's sufficient-statistics objective; this preserves the measured snapshot.

Defaults change to Newton, full-residue EM and pair batches of 128. Sparse
storage still requires explicit selection. Run the parent's installation/test
commands, then:

```sh
mkdir -p results
ci-mist -t TRAJECTORY.xtc -s TOPOLOGY.pdb -o results/newton --seed 0 \
  --probability_storage sparse
```

See the parent report for input/evidence availability and output-format
migration. Do not present Newton-versus-BFGS as a numerical-equivalence claim.
