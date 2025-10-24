# fisher-info-limits

authors: Carlo Paris, Steeve Laquitaine, Matthew Chalk & Ulisse Ferrari

Code for the paper on the limitations of the Fisher information metric

Tested on Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K ＠3.2 GHz/5.8 GHz)

Execution time: ~2 hours


# Description 

The dataset includes the firing rates of 41 Retina Ganglion Cells evoked by natural images (Golding et al., 2022):

```bash
* data/                         # (10 GB)
    * contrast_cells/           # (8.3GB) contains RGCs firing rate displayed natural image PCs
    * computed_contrast_cells/  # (709 MB) contains computed Bayes error and SSI
    * DS_cells.pkl              # (246 MB) data for quads of direction selective RGCs
    * bayesian_decoding_error/  # (916K) precomputed Bayes decoding error for quads of direction selective RGCs; uniform prior
    * decoding_analysis/        # (256 MB) parameters of the quads of RGCs tuning curves
```

# Instructions

- Run notebooks in notebook/