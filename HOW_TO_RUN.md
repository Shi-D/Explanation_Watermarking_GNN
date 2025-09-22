## Core Workflow

### 1. Train and Watermark Model

Train a GNN with embedded watermarks in graph substructures.

**Command:**
```bash
python main.py \
    --dataset_name photo \
    --subgraph_size_as_fraction 0.005 \
    --numSubgraphs 7 \
    --numLayers 3 \
    --hDim 256 \
    --epsilon 0.01 \
    --coefWmk 100 \
    --dropout 0.1 \
    --epochs 300 \
    --lr 0.0002 \
    --arch GCN \
    --seed 1 \
    --create_or_load_data create \
    --save_data \
    --preserve_edges_between_subsets
```

**Key Parameters:**
- `--dataset_name`: Dataset name (CS, photo, computers, etc.)
- `--subgraph_size_as_fraction`: Size of each watermarked subgraph as fraction of total nodes
- `--numSubgraphs`: Number of subgraphs to watermark
- `--arch`: GNN architecture (GCN, GAT, SAGE, SGC, Transformer)
- `--coefWmk`: Weight for watermark loss term
- `--epsilon`: Watermark loss epsilon parameter
- `--create_or_load_data`: Use "create" for new datasets, "load" for existing
- `--preserve_edges_between_subsets`: Always include this flag

**Note:** Run with multiple seeds (1, 2, 3, 4, 5) to collect multiple observations.

### 2. Collect Distribution Information

Generate statistical distributions for watermark detection significance testing.

**Prerequisites:** Must complete step 1 first.

**Command:**
```bash
python collect_random_subgraphs.py \
    --dataset_name photo \
    --subgraph_size_as_fraction 0.005 \
    --numSubgraphs 7 \
    --numLayers 3 \
    --hDim 256 \
    --epsilon 0.01 \
    --coefWmk 100 \
    --dropout 0.1 \
    --epochs 300 \
    --lr 0.0002 \
    --arch GCN \
    --seed 1 \
    --create_random_subgraphs \
    --get_count_list \
    --compute_regression_results \
    --get_match_distribution
```

**Important:** All hyperparameters must exactly match those used in training.

### 3. Test Statistical Significance

Compute significance of watermarking results across all seeds.

**Prerequisites:** Must complete steps 1 and 2 first.

**Command:**
```bash
python compute_results_significance.py \
    --dataset_name photo \
    --subgraph_size_as_fraction 0.005 \
    --numSubgraphs 7 \
    --numLayers 3 \
    --hDim 256 \
    --epsilon 0.01 \
    --coefWmk 100 \
    --dropout 0.1 \
    --epochs 300 \
    --lr 0.0002 \
    --arch GCN
```

**Note:** Do NOT specify `--seed` as this computes results across all seeds.

### 4. Test Robustness (Optional)

Evaluate watermark robustness against pruning and fine-tuning attacks.

**Prerequisites:** Must complete steps 1, 2, and 3 first.

**Command:**
```bash
python prune_and_fine_tune_en_masse.py \
    --dataset_name photo \
    --subgraph_size_as_fraction 0.005 \
    --numSubgraphs 7 \
    --numLayers 3 \
    --hDim 256 \
    --epsilon 0.01 \
    --coefWmk 100 \
    --dropout 0.1 \
    --epochs 300 \
    --lr_original 0.0002 \
    --arch GCN \
    --seed 1 \
    --prune \
    --fine_tune \
    --lr_scale 0.1
```

**Key Differences:**
- Use `--lr_original` instead of `--lr`
- Optional fine-tuning learning rate control:
  - `--lr_fine_tune X`: Set explicit fine-tuning learning rate
  - `--lr_scale X`: Set as fraction of original learning rate
  - If neither specified, uses original training learning rate

### 5. Generate Research Figures and Analysis

Use the Jupyter notebook for comprehensive analysis and publication-ready figures.

**Prerequisites:** Complete steps 1-4 for datasets and configurations of interest.

**Workflow:**
1. **Open Notebook**: Launch `paper_metrics_and_figures.ipynb`
2. **Configure Paths**: Set `root_folder` and `training_folder` paths to your results
3. **Run Analysis Sections**:
   - **Baseline Performance**: Compare classification-only accuracy across architectures
   - **Watermark Effectiveness**: Analyze backdoor trigger success rates
   - **Ablation Studies**: Generate subgraph size and count comparison plots
   - **Robustness Analysis**: Visualize attack resistance (pruning, fine-tuning)
   - **Theoretical Validation**: Plot subgraph overlap probabilities

**Key Analysis Functions:**
- `ablation_plot()`: Performance vs. hyperparameter analysis
- `plot_overlap_probability()`: Theoretical collision analysis  
- `plot_all_datasets()`: Cross-dataset robustness comparison
- `plot_all_datasets_ablation()`: Multi-dataset ablation studies

**Output:**
- Publication-ready PNG figures (300 DPI)
- Statistical tables in LaTeX format
- Comprehensive performance metrics
- P-value significance analysis

## Advanced Features

### Knowledge Distillation Attack

Test watermark resistance against knowledge distillation attacks.

**Command:**
```bash
python knowledge_distillation.py \
    --dataset_name photo \
    --subgraph_size_as_fraction 0.005 \
    --numSubgraphs 7 \
    --numLayers 3 \
    --hDim 256 \
    --epsilon 0.01 \
    --coefWmk 100 \
    --dropout 0.1 \
    --epochs 300 \
    --lr 0.0002 \
    --arch GCN \
    --seed 1 \
    --num_iters 5 \
    --KD_alpha 0.8 \
    --KD_temp 4.0 \
    --arch_student GCN \
    --numLayers_student 2 \
    --hDim_student 128 \
    --get_p_val
```

**Knowledge Distillation Parameters:**
- `--KD_alpha`: Balance between distillation and classification loss (0.0-1.0)
- `--KD_temp`: Temperature for softening teacher predictions (1.0-5.0)
- `--arch_student`, `--numLayers_student`, `--hDim_student`: Student model architecture (typically smaller)

**Training Modes:**
- `--kd_train_on_subgraphs`: Include watermarked subgraphs in training
- `--kd_subgraphs_only`: Train only on watermarked subgraphs

### Graph-Level Classification Watermarking

For graph-level classification tasks (e.g., molecular property prediction):

**Workflow:**
1. **Open Graph Classification Notebook**: Launch `watermark_graph_clf_level.ipynb`
2. **Dataset Selection**: Choose from TU Dataset collection or specify custom dataset
3. **Automated Setup**: Use built-in download and preprocessing functions
4. **Training Configuration**: Set collections, subgraph parameters, and model architecture
5. **Execute Training**: Run `train_with_watermark()` with monitoring
6. **Statistical Validation**: Built-in significance testing and visualization

**Example Usage:**
```python
# Download and preprocess MUTAG dataset
dataset_name = 'MUTAG'
url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip"
download_file(url, f"{dataset_name}.zip")

# Configure watermarking parameters
num_collections = 5
num_subgraphs_per_collection = 5
subgraph_size = 10

# Train with integrated watermarking and analysis
model, subgraph_collection_dict, primary_loss_curve, watermark_loss_curve, mu, sig, p = train_with_watermark(
    dataset_name, num_node_features, num_classes, 
    train_ratio, val_ratio, test_ratio, 
    subgraph_size=subgraph_size, 
    num_collections=num_collections, 
    seed=seed
)
```

## Parameter Reference

### Model Architecture
- `--arch`: GCN, GAT, SGC, SAGE, Transformer
- `--numLayers`: Number of GNN layers
- `--hDim`: Hidden dimension size
- `--dropout`: Dropout rate
- `--skip_connections`: Include residual connections

### Training
- `--lr`: Learning rate
- `--epochs`: Training epochs
- `--use_pcgrad`: Use PCGrad for conflicting gradients

### Watermarking
- `--watermark_type`: "unimportant" or "most_represented" features
- `--subgraph_method`: Subgraph extraction method (khop, random, rwr)
- `--percent_of_features_to_watermark`: Percentage of features to watermark

### Evaluation
- `--confidence`: Statistical confidence for watermark detection
- `--get_p_val`: Compute significance tests
- `--continuation`: Continue training from checkpoint
- `--starting_epoch`: Starting epoch for continuation

### Advanced Options
- `--sacrifice_method`: Set to "subgraph_node_indices" to exclude watermarked nodes from classification training
- `--preserve_edges_between_subsets`: Always include this flag

## Known Parameter Inconsistencies

Due to ongoing development, you may encounter these naming variations:
- `dataset` vs `dataset_name` (use `dataset_name`)
- `subgraph_size` vs `subgraph_size_as_fraction` (use `subgraph_size_as_fraction`)
- `nLayers` vs `numLayers` (use `numLayers`)
- `watermark_loss_epsilon` vs `epsilon` (use `epsilon`)

## Deprecated Features

The following features may appear in old code but should be avoided:
- SAM optimization (sam_rho, etc.)
- L2 regularization parameters
- regression_lambda (now hardcoded)

If you encounter errors related to these features, remove the relevant parameters from your commands.