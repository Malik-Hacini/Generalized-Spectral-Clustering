- Run full runtime benchmark
- Add lead_lag plots and explanation
REPRODUCIBILITY : 

- Add doc

[ ] rewrite readme
 
EXPLICIT REVIEWER REQUESTS :

Review 1 :

- [ ] real networks
- [ ] analysis on connecctivity sensitivity

  weak-connectedness needed : classical spectral methods result...
  not much more impact beyond that apart from computational cost (sparsity)

Review 2 :

- [ ] dataset examples of classical sc failing
- [ ] one large scale digraph (mnist64 ? )
- [ ] disbm
- [ ] detailed analysis of the computational complexity of the GSC method, including comparisons with existing spectral clustering techniques.

  done, same complexity as classical sc for single run.

Review 3:

- [ ] guidance for choosing nu
- [ ] tests on other simpler measures
- [ ] runtime and comparison to other baselines

