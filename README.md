# GNNBoundary Reproducibility Study

This repo contains the implementation of the reproducibility study of the paper "[GNNBoundary: Towards Explaining Graph Neural Networks Through the Lens of Decision Boundaries]" conducted at the University of Amsterdam.

## Reproducibility test files
The reproduction files of the core study are:
- `gnnboundary_collab.py`, `gnnboundary_motif.py`, `gnnboundary_enzymes.py` - which can each be executed from the provided Jupyter notebook `Main.ipynb` by calling the respective main function.

Executing these files produces the quantitative results from the study stored in logging files in the folder called `logs`. The corresponding figures are saved to the `figures` folder.

Furthermore, there are respective `gnninterpreter_{dataset}.py` files which incorporate GNNInterpreter from the authors' previous work titled ""GNNInterpreter: A Probabilistic Generative Model-Level Explanation for Graph Neural Networks". The core function of these files is to train and sample from GNNInterpreter as it is required for the analysis of GNNBoundary as outlined in the study. Note: Checkpoints for GNNInterpreter are provided under `Interpreter-ckpts` in case of slow convergence, as GNNInterpreter's performance is not the focus of this study.

The random baseline is retrieved by executing the cells of the notebook `random_baseline.ipynb`.

You must download the data from: https://drive.google.com/drive/folders/1s2y2dXTO_6oe8_bwsLpeL0ElOu4dLten and then drag and drop the `data` folder into the root directory of the project.

## Original paper extensions
We extend the original study by applying GNNBoundary to the PROTEINS dataset. The results for this dataset are produced via  `gnnboundary_proteins.py`, also callable in `Main.ipynb`. Furthermore, we apply principal component analysis on the boundary graph embeddings and visualize these along with the embeddings of the classifier datapoints in `pca_embeddings.ipynb`.

The implementation of Boundary Margin, Boundary Thickness and Boundary Complexity is found in `BoundaryTests.py`. 


## Environment and Installation

All reproducibility experiments have been conducted in an environment which can be installed using `FACT_environment.yml`. We are using Python 3.11.11, PyTorch 2.1.2 and PyTorch Geometric 2.5.3.

```bash
conda env create -f FACT_environment.yml
conda activate gnnboundaryFACT
```

## Changes from the original Repository

The original repository was modified, with additions such as:

1. Refactoring the original repository from Jupyter notebooks to Python files to run all experiments in a single run per dataset.
2. Incorporating GNNInterpreter into the codebase to train and sample graphs for every class.
3. Updated the .quantitative() method to calculate boundary margin, boundary thickness, boundary complexity given GNNInterpreter graphs as input.
4. Added .quantitativeInterpreter() method to allow for sampling with GNNInterpreter following training.
5. Implemented boundary margin, boundary thickness and boundary complexity computations.
6. Replaced draw_matrix() function with save_matrix() function to save figures and make them more presentable.
7. Addition of code to compute the random baseline, which was not found in the original repository.
8. Application of principal component analysis to the embeddings and corresponding visualization.
