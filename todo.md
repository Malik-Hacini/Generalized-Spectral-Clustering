EXPERIMENTS : 

- KNN - replace by gaussian weights
        - Numerical instabilities can arise (try to fix) + rbf +k knn is weird, and full rbf is illogical for the method, in addition to creating full matrices which makes memory complexity explode

- DiSBM : test and isolate directionality / asymmetry (github.com/SherylHYX/DIGRAC_Directed_Clustering/tree/main)
    - can generate datasets nicely, but still stuck with no gridesearch metric
- Real networks :

- 

MISC : 

- complexity : O(n^2 log n) , same for SC and GSC (w/o grid search) : BASICALLY DONE, cleanup the latex a bit.

- Add github link to paper preprint.


problem : no CH for gridsearch...  need an unsupervised metrics for real networks.

replacement ideas :
- Energy directly (i.e Rayleigh quotient)
- modularity