DSBM Models :

- GWENDAL generation
- Understand digrac generation
   - the different families
- Test digrac generation on graph_ch/modularity
   - gsc vs sc (and others also if interesting)
   - analyze correlation metric/ami


Real Networks :

- Fix lead lag data (nan or infinity for now)
- test lead lag datasets


Codebase :
Complexity : 
 - RUntimes for UCI with all baselines
  - design : add runtime in the pipeline + a plot script.
 - Optimize GSC grid search :
   - store graph computation for point clouds
     - other stuff could be done but very marginal
   - complexity analysis with this of grid_GSC vs SC
     - depends on the graph algorithm 
     - runtime exp
 
Codebase : 

- report ami max related to a metric max in experiment terminal output
- Refactor benchmarks in a folder... with clean imports (maybe a module ?)
