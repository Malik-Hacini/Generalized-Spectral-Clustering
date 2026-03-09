Real Networks :

- Fix lead lag data (nan or infinity for now)
- test lead lag datasets


Codebase :
Complexity : 
 - RUntimes for UCI with all baselines
  - design : add runtime in the pipeline + a plot script.
 - Optimize GSC grid search :
     - other stuff could be done but very marginal
       - could cache the measure P^t results   
        - would require to refactor parallelization logic for marginal improvements
   - complexity analysis with this of grid_GSC vs SC
     - depends on the graph algorithm 
     - runtime exp
 
- update readme with new metrics and explanation of the method
- Refactor benchmarks in a folder... with clean imports (maybe a module ?)
