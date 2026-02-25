# DIGRAC Data Import

This folder contains utilities to import DIGRAC datasets into the project graph format.

## Script

`import_digrac_data.py`

Imports from a cloned DIGRAC repository (`<repo>/data`) into:

- `DSBM_datasets/digrac/` for synthetic DSBM pickles (`*.pk`)
- `datasets/digrac_directed/` for real directed networks (`adj.npz`)

For real directed datasets, the importer keeps only datasets with label files
(`labels.npy`) and skips unlabeled ones.

Each imported dataset is saved as:

```
<dataset_dir>/graph.npz
```

with keys:

- `adj_data`, `adj_indices`, `adj_indptr`, `adj_shape`
- `labels` (only when available)

## Default command

```bash
source .venv/bin/activate
python data_import/import_digrac_data.py \
  --digrac-root /tmp/DIGRAC_Directed_Clustering \
  --output-dsbm-root DSBM_datasets/digrac \
  --output-real-root datasets/digrac_directed
```

## Output manifests

The script writes import manifests to:

- `DSBM_datasets/digrac/import_manifest.json`
- `datasets/digrac_directed/import_manifest.json`

These include counts, source files, node/edge stats, and label availability.
