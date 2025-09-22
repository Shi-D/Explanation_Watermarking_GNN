# Watermarking Graph Neural Networks via Explanations for Ownership Protection

This is the repo for paper 'Watermarking Graph Neural Networks via Explanations for Ownership Protection'.

## 1. Installation

Install dependencies from the requirements file. It's recommended to start with a conda environment:

```bash
conda create --name myenv --file requirements.txt
```

Alternatively, install directly to your current environment:

```bash
conda install --file requirements.txt
```

## 2. Additional Package Installation

One necessary package requires manual installation. Run these commands from your command line:

```bash
git clone https://github.com/WeiChengTseng/Pytorch-PCGrad.git
mv Pytorch-PCGrad pcgrad
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
mv pcgrad "$SITE_PACKAGES"
```

## 3. Datasets

### Node-Classification Level
For watermarking GNNs at the node-classification level, we work with three datasets:
- **Photo**
- **CS** 
- **PubMed**

*Note: These datasets each consist of a single graph.*

### Graph-Classification Level
For watermarking GNNs at the graph-classification level, we have tested:
- **MUTAG**

*Note: This dataset consists of multiple graphs.*

### Using Other Datasets
- See `dataset_attributes` within `config.py` for datasets currently recognized by the code
- To use a new dataset, add a new entry to this dictionary including:
  - The `torch_geometric.datasets` class it comes from
  - Additional information depending on whether it's node-classification or graph-classification
  - Number of classes, number of features, etc.

## 4. Repository Description

The file `REPO_FILE_INFO.md` outlines the repository structure, including important files and their main contents.

## 5. Execution Instructions

The file `HOW_TO_RUN.md` contains comprehensive information on running various files, including:
- Main watermarking pipeline
- Additional tasks like pruning, fine-tuning, knowledge distillation
- Alternate watermarking methods

**Note:** Instructions for graph classification-level watermarking can be found in `watermark_graph_clf_level.ipynb` instead.