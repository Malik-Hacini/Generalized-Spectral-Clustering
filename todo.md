Codebase :
Complexity :
- [ ] RUntimes for UCI with all baselines
  caveat : parallelization makes it hard to acutally compare + depends HEAVILY on the problem.
  plots are crowded (heatmap maybe ?) and it really is not an apples to apples comparison due to the different grid size, different optimization etc etc. check for sourced complexity of the baselines
- [ ] weak-connectedness of the UCI graphs
   - not all of them are weak connected, precise which
- [ ] update readme with new metrics and explanation of the method
- [ ] Refactor benchmarks in a folder... with clean imports (maybe a module ?)


PLOTS : 


Graph viz / Dirichlet energy failing : script to add
UCI : 
- HEATMAPS : rewrite the script
- Runtimes :
    - All baselines : to run
    - Single run : to write
- In degree distribtuion : to rewrite

SYNTHETIC : 

- Cluster imbalance :  script to add

 DiSBM : 
   - Complexity comparison : to clean and rerun
   - Graph_CH filter : todo


EXPLICIT REVIEWER REQUESTS :

Review 1 :

- [ ] real networks
- [ ] analysis on connecctivity sensitivity

  weak-connectedness needed : classical spectral methods result...
  not much more impact beyond that apart from computational cost (sparsity)

Review 2 :

- [ ] dataset examples of classical sc failing
 [ ] one large scale digraph (mnist64 ? )
- [ ] disbm
- [ ] detailed analysis of the computational complexity of the GSC method, including comparisons with existing spectral clustering techniques.

  done, same complexity as classical sc for single run.

Review 3:

- [ ] guidance for choosing nu
- [ ] tests on other simpler measures
- [ ] runtime and comparison to other baselines

