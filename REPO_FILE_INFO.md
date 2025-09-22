# Graph Neural Network Watermarking Repository

This repository implements watermarking techniques for Graph Neural Networks (GNNs) with tools for training, evaluation, and robustness testing.

The respository takes the following structure:

|--- ROOT
      |--- data 
      |--- training_results
      |--- src
      |     |--- backdoor_based_watermarking.py
      |     |--- collect_random_subgraphs.py
      |     |--- compute_results_significance.py
      |     |--- config.py
      |     |--- data_utils.py
      |     |--- embed_and_verify.py
      |     |--- general_utils.py
      |     |--- knowledge_distillation_utils.py
      |     |--- knowledge_distillation.py
      |     |--- main.py
      |     |--- models.py
      |     |--- paper_metrics_and_figures.ipynb
      |     |--- prune_and_fine_tune_en_masse.py
      |     |--- prune_and_fine_tune_utils.py
      |     |--- regression_utils.py
      |     |--- subgraph_utils.py
      |     |--- transform_functions.py
      |     |--- watermark_graph_clf_level.ipynb
      |     |--- watermark_graphlime_backdoor_pre_explain.py
      |     |--- watermark_utils.py
      |
      |--- requirements.txt
      |--- README.md
      |--- HOW_TO_RUN.md
      |--- REPO_FILE_INFO.md


## Repository Architecture

### Main Execution (`main.py`)
Primary command-line interface for watermarked GNN training with comprehensive argument parsing:

**Core Functionality**
- Unified entry point supporting all watermarking methods (regression-based, random backdoor, GraphLIME backdoor)
- Extensive command-line argument parsing for all configuration parameters
- Multi-iteration experimental loops with automatic seed management
- Automatic watermark size recommendation based on statistical confidence levels

**Key Features**
- **Dataset Management**: Automatic data loading/creation with configurable train/val/test splits
- **Architecture Configuration**: Full GNN architecture customization (GAT, GCN, SAGE, SGC, Transformer)
- **Watermarking Strategy Selection**: Support for all three watermarking approaches with method-specific parameters
- **Optimization Settings**: Learning rate, epochs, PCGrad, dropout, augmentation controls
- **Statistical Analysis**: Automatic computation of recommended watermark length based on natural distribution analysis
- **Results Management**: Structured result logging with performance metrics and significance testing

**Usage Modes**
- Regression-based watermarking (default)
- Classification-only training (`--clf_only`)
- Random backdoor watermarking (`--watermark_random_backdoor`) 
- GraphLIME backdoor watermarking (`--watermark_graphlime_backdoor`)

### Configuration System (`config.py`)
- Dataset-specific hyperparameter presets
- Watermarking strategy configuration  
- Training optimization settings
- Architecture specifications
- Attack and defense parameters

### Main Training and Watermarking Functions (`embed_and_verify.py`)
The central orchestrator for watermarked GNN training:

**`Trainer` Class**
- Manages end-to-end training with watermark embedding
- Handles data augmentation, loss computation, and evaluation
- Supports multiple training modes: classification-only, watermarked training, and backdoor attacks
- Key methods:
  - `train()`: Main training loop with watermark application
  - `closure_watermark()`: Computes combined classification + watermark losses
  - `closure_random_backdoor_watermark()`: Handles random trigger backdoor training
  - `closure_graphlime_backdoor_watermark()`: Handles GraphLIME-based backdoor training
  - `apply_watermark_()`: Embeds watermarks into subgraph features
  - `get_watermark_performance_()`: Evaluates watermark detection strength

**Key Functions**
- `accuracy()`: Classification accuracy computation
- `setup_history()`: Initialize training metrics tracking
- `get_watermark_performance()`: Statistical watermark detection analysis
- `separate_forward_passes_per_subgraph()`: Isolated subgraph evaluation
- `gather_random_subgraphs_not_wmk()`: Generate control subgraphs for statistical testing

### Neural Network Models (`models.py`)
Unified GNN architecture supporting multiple frameworks:

**`Net` Class**
- Supports GCN, GAT, SAGE, SGC, Transformer architectures
- Configurable layers, dimensions, dropout, skip connections
- Integrated batching support for large graphs via `DataLoaderRegistry`
- Automatic architecture selection based on configuration

**`DataLoaderRegistry`**
- Manages memory-efficient batched processing
- Handles different sampling strategies for train/eval modes
- Persistent loader caching with pickle-safe serialization

#### Subgraph Extraction (`subgraph_utils.py`)
Core subgraph generation and manipulation utilities:

**Key Functions**
- `generate_subgraph()`: Main subgraph extraction interface supporting multiple methods
- `create_khop_subgraph()`: K-hop neighborhood extraction around central nodes
- `create_random_subgraph()`: Random node sampling with mask/avoidance support
- `create_rwr_subgraph()`: Random walk with restart subgraph generation
- `rank_training_nodes_by_degree()`: Identify high-degree nodes for k-hop centering
- `get_masked_subgraph_nodes()`: Extract subgraphs respecting train/val/test masks

#### Data Management (`data_utils.py`)
Dataset loading, preprocessing, and augmentation:

**Key Functions**
- `prep_data()`: Unified dataset loading with train/val/test splitting
- `augment_data()`: Graph data augmentation (node drop, mixup, feature masking, edge drop)
- `get_subgraph_from_node_indices()`: Extract subgraphs while preserving data structure
- `get_classification_train_nodes()`: Handle node sacrifice for watermark isolation
- `initialize_experiment_data()`: Setup datasets for experiments

### Transform Functions (`transform_functions.py`)
Comprehensive data preprocessing and augmentation utilities:

**Data Transformations**
- `DensifyTransform`: Convert sparse features to dense using Fourier transforms or noise injection
- `CreateMaskTransform`: Generate train/validation/test splits with configurable ratios
- `GraphFourierTransform`: Apply graph Fourier analysis for spectral domain processing
- `KHopsFractionDatasetTransform`: Extract k-hop subgraphs as dataset fractions
- `SparseToDenseTransform`: Convert sparse tensor formats to dense

**Mask Management**
- `ChooseLargestMaskForTrain`: Automatically select largest available split for training
- `CreateMasksInOrder`: Systematic mask generation with deterministic ordering
- Support for both random and deterministic split generation

**Preprocessing Pipeline Integration**
- Compatible with PyTorch Geometric data loading pipelines
- Seed-controlled reproducible transformations
- Memory-efficient processing for large graphs

### Watermarking Utilities (`watermark_utils.py`)
Core watermark creation and management:

**Key Functions**
- `collect_subgraphs_within_single_graph_for_watermarking()`: Extract subgraphs for watermarking
- `apply_watermark()`: Create and embed watermarks using different strategies
- `create_watermarks_at_most_represented_indices()`: Target frequently-occurring features
- `collect_watermark_values()`: Generate balanced +1/-1 watermark patterns
- `create_basic_watermarks()`: Random feature watermarking

### Statistical Analysis (`regression_utils.py`)
Watermark detection through regression analysis:

**Key Functions**
- `solve_regression()`: Core regression solver using Gaussian kernels
- `compute_kernel()`: Gaussian (RBF) kernel computation with normalization
- `compute_gram_matrix()`: Centered and normalized Gram matrix computation
- `regress_on_subgraph()`: Apply regression analysis to subgraph features

### Utility Functions (`general_utils.py`)
System-wide utilities and configuration management:

**Result Management**
- `save_results()`: Persist models, histories, and configurations
- `get_results_folder_name()`: Generate organized directory structures
- `merge_kwargs_dicts()`: Combine configuration dictionaries

**Naming and Tagging**
- `get_model_tag()`: Generate model configuration identifiers
- `get_watermark_tag()`: Create watermark configuration strings
- `get_subgraph_tag()`: Generate subgraph method identifiers

**Statistical Utilities**
- `count_matches()`: Count feature sign matches across subgraphs
- `count_beta_matches()`: Statistical distribution analysis for significance testing
- `replace_history_Nones()`: Clean training history data

### Paper Metrics and Figures (`paper_metrics_and_figures.ipynb`)
Comprehensive Jupyter notebook for research analysis, visualization, and paper figure generation:

**Performance Analysis Functions**
- Performance comparison across datasets (Photo, CS, PubMed) and architectures (SAGE, GCN, SGC)
- Baseline accuracy analysis for classification-only models
- Watermark effectiveness evaluation for backdoor-based approaches (random and GraphLIME triggers)
- Statistical significance testing and p-value computation

**Ablation Study Visualization**
- `ablation_plot()`: Generate performance vs. hyperparameter plots
- Subgraph size impact analysis (varying from 0.001 to 0.01 fraction of nodes)
- Number of watermarked subgraphs impact analysis (2-5 subgraphs)
- Multi-dataset comparison with consistent formatting

**Robustness Analysis**
- `gather_prune_stats()`: Extract pruning attack results (structured/unstructured)
- `gather_fine_tune_stats()`: Analyze fine-tuning attack effectiveness
- `prune_plot()` and `fine_tune_plot()`: Visualize watermark persistence under attacks
- Statistical significance tracking through attacks

**Theoretical Analysis**
- `plot_overlap_probability()`: Subgraph overlap probability analysis
- Binomial distribution modeling for watermark collision analysis
- Mathematical validation of watermarking approach assumptions

**Figure Generation Pipeline**
- Publication-ready matplotlib figures with consistent styling
- Multi-panel layouts for comprehensive comparisons
- Automated data collection from training result directories
- Configurable aesthetics (font sizes, colors, DPI settings)

**Key Visualization Functions**
- `plot_all_datasets()`: Cross-dataset comparison plots
- `plot_all_datasets_ablation()`: Ablation study visualizations
- `plot_all_fine_tune_in_grid()` and `plot_all_prune_in_grid()`: Attack analysis grids
- Statistical arrow annotations for off-plot values
- P-value visualization with significance thresholds

**Research Workflow Integration**
- Automatic result parsing from training output files
- Dataset-specific hyperparameter configuration dictionaries
- Seed aggregation for statistical robustness
- LaTeX table generation for paper integration

The notebook serves as the primary tool for generating all research figures and conducting post-training analysis, with built-in support for the three main experimental datasets and all GNN architectures used in the research.

### `collect_random_subgraphs.py` - Generate Null Distribution
Creates the statistical baseline for significance testing:

**Key Functions:**
- Loads trained watermarked models
- Generates random (non-watermarked) subgraphs of same size
- Applies regression analysis to random subgraphs  
- Computes natural distribution parameters (μ, σ) for match counts
- Saves regression coefficients and distribution statistics

**Process:**
1. **Model Loading**: Load trained model matching experiment hyperparameters
2. **Random Subgraph Generation**: Create `num_iters` random subgraphs via `gather_random_subgraphs_not_wmk()`
3. **Regression Analysis**: Apply kernel regression to each random subgraph
4. **Distribution Fitting**: Compute μ_natural and σ_natural from 1000 random samplings
5. **Persistence**: Save raw regression results and distribution parameters

**Output Files:**
- `raw_betas_list_size_{fraction}.pkl`: Raw regression coefficients
- `distribution.txt`: Natural distribution parameters (μ, σ)
- `distribution_values_{dataset}_all_sizes.txt`: Detailed statistics

### `compute_results_significance.py` - Statistical Significance Testing
Computes p-values for watermark detection across multiple seeds:

**Key Functions:**
- Loads training results and natural distributions
- Extracts watermark match counts from training logs
- Computes Z-scores and p-values for each experimental run
- Aggregates significance across multiple seeds

**Process:**
1. **Result Extraction**: Parse training logs for watermark match counts
2. **Distribution Loading**: Load μ_natural and σ_natural from previous step  
3. **Significance Calculation**: 
   - Z-score = (training_matches - μ_natural) / σ_natural
   - p-value = 1 - Φ(Z-score) where Φ is cumulative normal distribution
4. **Multi-Seed Aggregation**: Average significance across all seeds

**Output:**
- Individual seed significance in `results_actual_significance.txt`
- Aggregated significance across all seeds
- Z-scores and p-values for statistical reporting

## Robustness Testing

### Batch Attack Testing (`prune_and_fine_tune_en_masse.py`)
Automated robustness evaluation against pruning and fine-tuning attacks:

**Attack Methods**
- **Structured Pruning**: Remove entire neurons/channels systematically
- **Unstructured Pruning**: Remove individual weights based on magnitude
- **Fine-tuning**: Continued training on clean data to remove watermarks

**Key Functions**
- Loads pre-trained watermarked models from training results
- Applies attacks with varying intensities and hyperparameters
- Evaluates watermark persistence through statistical significance testing
- Generates comprehensive attack analysis reports

**Configuration**
- Target confidence levels for significance testing
- Learning rate scaling for fine-tuning attacks
- Model architecture and dataset-specific parameter loading
- Continuation support for interrupted experiments

### Attack Utilities (`prune_and_fine_tune_utils.py`)
Supporting utilities for implementing pruning and fine-tuning attacks:

**Pruning Implementation**
- `run_prune()`: Execute structured/unstructured pruning with magnitude-based weight removal
- Pytorch-native pruning support with configurable sparsity levels
- Watermark evaluation after each pruning iteration

**Fine-tuning Implementation** 
- `run_fine_tune()`: Implement fine-tuning attacks with reduced learning rates
- Clean data training to overwrite watermark patterns
- Early stopping based on watermark degradation

## Additional Watermarking Methods

### Backdoor Watermarking (`backdoor_based_watermarking.py`)
Alternative watermarking approaches using backdoor triggers:

**Random Trigger Backdoors**
- `create_random_trigger_graph()`: Generate random trigger subgraphs
- `inject_backdoor_trigger_subgraph()`: Insert triggers into existing graphs

**GraphLIME Backdoors** 
- `backdoor_GraphLIME()`: Apply GraphLIME-based feature ranking for backdoors
- `get_ranked_features_GraphLIME()`: Rank features by importance for targeted modification

### GraphLIME Backdoor Preprocessing (`watermark_graphlime_backdoor_pre_explain.py`)
Specialized preprocessing for GraphLIME-based backdoor watermarking:

**GraphLIME Integration**
- Implements feature importance ranking using GraphLIME explanations
- Identifies most influential node features for targeted backdoor placement
- Supports explanation-guided trigger generation

**Preprocessing Pipeline**
- Loads datasets and applies GraphLIME analysis
- Generates ranked feature importance lists
- Prepares data structures for GraphLIME backdoor training
- Maintains compatibility with main training pipeline

**Configuration Support**
- Command-line interface matching main training script
- GraphLIME-specific hyperparameter management
- Poisoning rate and target label configuration

### Graph-Level Classification Watermarking (`watermark_graph_clf_level.ipynb`)
Specialized notebook implementing watermarking for graph-level classification tasks with external dataset support:

**Core Functionality**
- Graph-level classification watermarking (vs. node-level in main framework)
- External dataset integration from TU Dataset collection
- Automated dataset downloading and preprocessing
- Custom graph neural network architecture for graph classification

**Key Functions and Classes**
- `download_file()` and `unzip_file()`: Automated dataset acquisition from online repositories
- `create_dataset_from_files()`: Parse raw TU Dataset format into PyTorch Geometric Data objects
- `split_dataset()`: Create train/validation/test splits for graph-level tasks
- `generate_subgraph_graph_clf()`: Extract subgraphs from individual graphs for watermarking
- `build_subgraph_collections()`: Create collections of watermarked subgraphs across multiple graphs
- `graph_clf_model`: Custom GNN architecture with global pooling for graph-level predictions

**Watermarking Strategy for Graph Classification**
- **Subgraph Collection Approach**: Instead of watermarking within a single large graph, creates collections of subgraphs sampled from multiple training graphs
- **Feature Aggregation Methods**: Support for 'flatten', 'average', and 'sum' aggregation of node features within subgraphs
- **Cross-Graph Watermarking**: Embeds coordinated watermarks across subgraphs from different source graphs
- **Statistical Detection**: Uses same regression-based approach but adapted for graph-level prediction contexts

**Training Pipeline**
- `train_with_watermark()`: Complete training workflow with integrated watermarking
- Dual-loss optimization: graph classification + watermark embedding losses
- PCGrad support for handling conflicting gradients
- Real-time watermark alignment monitoring during training

**Dataset Integration**
- **TU Dataset Support**: Direct integration with TU Dortmund graph dataset collection
- **Format Compatibility**: Handles standard graph kernel dataset formats (edge lists, node labels, graph indicators)
- **Preprocessing Pipeline**: Automatic conversion to PyTorch Geometric format with one-hot encoding
- **Popular Datasets**: MUTAG, PROTEINS, DD, NCI1, and other molecular/social network datasets

**Statistical Validation**
- Built-in significance testing with null distribution generation
- Random subgraph baseline comparison for statistical confidence
- P-value computation for ownership verification
- Multi-seed experimental design for robustness

**Example Usage Workflow**
1. **Dataset Selection**: Choose from TU Dataset collection (MUTAG, PROTEINS, etc.)
2. **Automatic Download**: Fetch and preprocess dataset from online repository
3. **Hyperparameter Configuration**: Set number of collections, subgraphs per collection, and sizes
4. **Training Execution**: Run watermarked training with real-time monitoring
5. **Statistical Analysis**: Generate null distributions and compute significance
6. **Visualization**: Plot training curves and watermark performance metrics

**Experimental Design Support**
- Multi-seed experimental loops for statistical rigor
- Hyperparameter sweeps (subgraph sizes, collection counts)
- Automated result persistence and organization
- Loss curve visualization and performance tracking

**Research Applications**
- Molecular property prediction watermarking
- Social network analysis with ownership protection
- Bioinformatics graph classification with IP protection
- Comparative studies between node-level and graph-level watermarking approaches

This notebook extends the main watermarking framework to graph-level classification tasks, enabling watermark embedding in scenarios where the prediction target is the entire graph rather than individual nodes. It provides a complete pipeline from dataset acquisition to statistical validation, making it particularly useful for molecular and social network applications.

## Knowledge Distillation

### Main Knowledge Distillation (`knowledge_distillation.py`)
Implementation of knowledge distillation as a defense against watermarking:

**Attack Strategy**
- Train smaller "student" networks to mimic watermarked "teacher" models
- Attempt to transfer classification performance without watermarks
- Test whether watermark patterns survive distillation process

**Key Features**
- **Teacher-Student Architecture**: Configurable teacher and student GNN architectures
- **Distillation Loss**: Temperature-scaled knowledge transfer with KL divergence
- **Training Modes**: Full dataset distillation vs. subgraph-only distillation
- **Statistical Validation**: P-value computation for watermark detection post-distillation

### Knowledge Distillation Utilities (`knowledge_distillation_utils.py`)
Core implementation of the `Trainer_KD` class for knowledge distillation:

**`Trainer_KD` Class**
- Manages teacher-student training with combined classification and distillation losses
- Implements temperature-scaled softmax for knowledge transfer
- Supports PCGrad for handling conflicting gradient directions
- Provides watermark detection analysis throughout distillation process

**Key Methods**
- `distillation_loss()`: KL divergence between teacher and student predictions
- `closure_KD()`: Combined loss optimization with configurable α weighting
- `test_watermark()`: Statistical significance testing for watermark persistence
- `train_KD()`: Main distillation training loop with real-time monitoring