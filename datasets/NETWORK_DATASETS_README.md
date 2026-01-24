# Real-World Network Benchmark Datasets

This document describes the labeled network datasets available for spectral clustering benchmarks. All datasets are stored in `graph.npz` format compatible with `load_dataset()`.

## Dataset Overview

| Dataset | Nodes | Edges | Classes | Type | Best For |
|---------|-------|-------|---------|------|----------|
| email_eu_core | 1,005 | 25,571 | 42 | Directed | Many-cluster community detection |
| polblogs | 1,490 | 19,025 | 2 | Directed | Binary classification, political polarization |
| football | 115 | 1,226 | 12 | Undirected | Multi-class sports conference detection |
| dolphins | 62 | 318 | 2 | Undirected | Small network bisection |
| karate | 34 | 156 | 2 | Undirected | Classic benchmark, fission detection |
| polbooks | 105 | 882 | 3 | Undirected | Political leaning classification |
| wiki_vote | 7,115 | 103,689 | 5 | Directed | Large-scale directed community detection |

---

## Dataset Details

### 1. Email-Eu-Core (`email_eu_core`)

**Description:**  
A directed email communication network from a large European research institution. Nodes represent members of the institution and edges represent email exchanges. Ground-truth labels correspond to the 42 departments of the institution.

**Statistics:**
- Nodes: 1,005
- Edges: 25,571 (directed)
- Classes: 42 (departments)
- Average degree: ~25.4

**Source:**  
SNAP - Stanford Large Network Dataset Collection  
https://snap.stanford.edu/data/email-Eu-core.html

**Citations:**
```bibtex
@inproceedings{yin2017local,
  title={Local Higher-Order Graph Clustering},
  author={Yin, Hao and Benson, Austin R and Leskovec, Jure and Gleich, David F},
  booktitle={Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={555--564},
  year={2017}
}

@article{leskovec2007graph,
  title={Graph evolution: Densification and shrinking diameters},
  author={Leskovec, Jure and Kleinberg, Jon and Faloutsos, Christos},
  journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume={1},
  number={1},
  pages={2--es},
  year={2007}
}
```

**Notes:**
- Suitable for testing GSC on directed networks with many communities
- Large number of classes (42) tests scalability of cluster detection

---

### 2. Political Blogs (`polblogs`)

**Description:**  
A directed network of hyperlinks between political blogs around the time of the 2004 U.S. presidential election. Nodes represent blogs, and edges represent hyperlinks between them. Labels indicate political leaning: 0 = liberal, 1 = conservative.

**Statistics:**
- Nodes: 1,490
- Edges: 19,025 (directed)
- Classes: 2 (liberal, conservative)
- Average degree: ~12.8

**Source:**  
Mark Newman's Network Data Repository  
https://public.websites.umich.edu/~mejn/netdata/

**Citation:**
```bibtex
@inproceedings{adamic2005political,
  title={The political blogosphere and the 2004 U.S. election: divided they blog},
  author={Adamic, Lila A and Glance, Natalie},
  booktitle={Proceedings of the 3rd International Workshop on Link Discovery},
  pages={36--43},
  year={2005}
}
```

**Notes:**
- Classic benchmark for detecting political polarization
- Directed network with strong community structure
- Tests binary classification with asymmetric link patterns

---

### 3. American College Football (`football`)

**Description:**  
Network of American football games between Division IA colleges during the regular season of Fall 2000. Nodes represent teams, and edges represent games played between them. Labels correspond to the 12 athletic conferences.

**Statistics:**
- Nodes: 115
- Edges: 613 undirected (1,226 when counted as directed)
- Classes: 12 (conferences)
- Average degree: ~10.7

**Source:**  
Mark Newman's Network Data Repository  
https://public.websites.umich.edu/~mejn/netdata/

**Citation:**
```bibtex
@article{girvan2002community,
  title={Community structure in social and biological networks},
  author={Girvan, Michelle and Newman, Mark EJ},
  journal={Proceedings of the National Academy of Sciences},
  volume={99},
  number={12},
  pages={7821--7826},
  year={2002}
}
```

**Notes:**
- Classic community detection benchmark
- Well-defined ground truth (athletic conferences)
- Good for testing multi-class clustering algorithms
- Teams preferentially play within their conference

---

### 4. Dolphins Social Network (`dolphins`)

**Description:**  
An undirected social network of frequent associations between 62 dolphins living in Doubtful Sound, New Zealand. The community labels are derived from spectral analysis, reflecting the known fission of the dolphin community into two groups.

**Statistics:**
- Nodes: 62
- Edges: 159 undirected (318 when counted as bidirectional)
- Classes: 2 (derived from fission event)
- Average degree: ~5.1

**Source:**  
Mark Newman's Network Data Repository  
https://public.websites.umich.edu/~mejn/netdata/

**Citation:**
```bibtex
@article{lusseau2003bottlenose,
  title={The bottlenose dolphin community of Doubtful Sound features a large proportion of long-lasting associations},
  author={Lusseau, David and Schneider, Karsten and Boisseau, Oliver J and Haase, Patti and Slooten, Elisabeth and Dawson, Steve M},
  journal={Behavioral Ecology and Sociobiology},
  volume={54},
  number={4},
  pages={396--405},
  year={2003}
}
```

**Notes:**
- Small network good for visualization and detailed analysis
- Natural binary split in the community
- Tests algorithm behavior on small-scale networks

---

### 5. Zachary's Karate Club (`karate`)

**Description:**  
The canonical social network benchmark: friendships among 34 members of a karate club at a US university in the 1970s. The club split into two factions after a dispute between the instructor (Mr. Hi) and the club president (Officer), providing ground-truth labels.

**Statistics:**
- Nodes: 34
- Edges: 78 undirected (156 when counted as bidirectional)
- Classes: 2 (Mr. Hi's group, Officer's group)
- Average degree: ~4.6

**Source:**  
Mark Newman's Network Data Repository  
https://public.websites.umich.edu/~mejn/netdata/

**Citation:**
```bibtex
@article{zachary1977information,
  title={An information flow model for conflict and fission in small groups},
  author={Zachary, Wayne W},
  journal={Journal of Anthropological Research},
  volume={33},
  number={4},
  pages={452--473},
  year={1977}
}
```

**Notes:**
- The most widely-used community detection benchmark
- Essential for comparing with published results
- Ground-truth from real social fission event

---

### 6. Political Books (`polbooks`)

**Description:**  
A network of books about US politics published around the time of the 2004 presidential election. Edges represent frequent co-purchasing by Amazon buyers. Labels indicate political leaning: liberal (l=0), neutral (n=1), conservative (c=2).

**Statistics:**
- Nodes: 105
- Edges: 441 undirected (882 when counted as bidirectional)
- Classes: 3 (liberal, neutral, conservative)
- Average degree: ~8.4

**Source:**  
Mark Newman's Network Data Repository (originally from V. Krebs)  
https://public.websites.umich.edu/~mejn/netdata/

**Citation:**
```bibtex
@misc{krebs2004polbooks,
  author = {Krebs, Valdis},
  title = {Political Books Network},
  year = {2004},
  note = {Available at www.orgnet.com}
}
```

**Notes:**
- Three-way classification (liberal, neutral, conservative)
- Tests detection of intermediate/bridge communities
- Co-purchase patterns reflect ideological similarity

---

### 7. Wikipedia Vote Network (`wiki_vote`)

**Description:**  
A directed network of voting interactions between Wikipedia users during admin elections. A directed edge from user A to user B means user A voted on user B's admin nomination. Community labels are derived from spectral clustering (5 clusters).

**Statistics:**
- Nodes: 7,115
- Edges: 103,689 (directed)
- Classes: 5 (derived from network structure)
- Average degree: ~14.6

**Source:**  
SNAP - Stanford Large Network Dataset Collection  
https://snap.stanford.edu/data/wiki-Vote.html

**Citation:**
```bibtex
@inproceedings{leskovec2010predicting,
  title={Predicting positive and negative links in online social networks},
  author={Leskovec, Jure and Huttenlocher, Daniel and Kleinberg, Jon},
  booktitle={Proceedings of the 19th International Conference on World Wide Web},
  pages={641--650},
  year={2010}
}
```

**Notes:**
- Largest dataset in this collection
- Tests scalability on larger networks
- Directed voting patterns reveal community structure
- Labels derived from spectral analysis (no ground truth)

---

## Usage Example

```python
from utils.file_manager import load_dataset

# Load a graph dataset
adj_matrix, labels = load_dataset('datasets', 'polblogs')

print(f"Shape: {adj_matrix.shape}")
print(f"Edges: {adj_matrix.nnz}")
print(f"Classes: {len(set(labels))}")
print(f"Is directed: {(adj_matrix != adj_matrix.T).nnz > 0}")
```

## Suitability for Generalized Spectral Clustering

These datasets are particularly well-suited for GSC evaluation because:

1. **Directed networks** (`email_eu_core`, `polblogs`, `wiki_vote`): Test the key advantage of GSC—handling asymmetric adjacency matrices without symmetrization that loses directional information.

2. **Ground-truth labels**: Enable quantitative evaluation using NMI, ARI, and other clustering metrics.

3. **Varied scales**: Range from 34 to 7,115 nodes, testing algorithm behavior across scales.

4. **Multiple cluster counts**: From 2 to 42 classes, testing single-k to many-k clustering.

5. **Real-world structure**: Networks exhibit realistic properties (degree distributions, clustering coefficients) unlike synthetic benchmarks.
