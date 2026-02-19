EXPERIMENTS : 
- DiSBM : test and isolate directionality / asymmetry (github.com/SherylHYX/DIGRAC_Directed_Clustering/tree/main)
    - can generate datasets nicely, but still stuck with no gridesearch metric
- Real networks :
- 
problem : no CH for gridsearch...  need an unsupervised metrics for real networks.

replacement ideas :
- modularity - testing
- ch on transition matrix (multiple iterations --> P^k) 
 - fix build_transition_matrix functiion in graph_ch to use the real markov chain we define
 - do more testing
