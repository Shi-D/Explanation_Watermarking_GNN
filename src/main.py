from embed_and_verify import *
from config import *
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Watermarking settings')
    parser.add_argument('--dataset_name', type=str, default='computers',help='Dataset Name')
    parser.add_argument('--seed',                                             type=int,               default=0,  help='Random seed.')
    parser.add_argument('--create_or_load_data',    type=str,               default='load',  help='Whether to build dataset from scratch or load it -- either "create" or "load".')
    parser.add_argument('--save_data',    action='store_true',               default=False,  help='Whether to build dataset from scratch or load it -- either "create" or "load".')

    args, _ = parser.parse_known_args()
    dataset_name = args.dataset_name
    seed = args.seed
    assert args.create_or_load_data in ['create','load']
    load_data = True if args.create_or_load_data=="load" else False
    save_data = args.save_data

    
    parser.add_argument('--train_ratio',      type=float,   default=dataset_attributes[dataset_name]['train_ratio'],  help='Ratio of dataset comprising train set.')
    args, _ = parser.parse_known_args()
    val_ratio = test_ratio = (1-args.train_ratio)/2
    if dataset_attributes[dataset_name]['single_or_multi_graph']=='single':
        dataset = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default',  train_val_test_split=[args.train_ratio,val_ratio,test_ratio], seed=seed, save=save_data, load=load_data)
        data = dataset[0]
    elif dataset_attributes[dataset_name]['single_or_multi_graph']=='multi':
        [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader] = prep_data(dataset_name=dataset_name, location='default', batch_size='default', transform_list='default', train_val_test_split=[args.train_ratio,val_ratio,test_ratio])

    print('data:',data)
        
    get_presets(dataset,dataset_name)

    parser.add_argument('--num_iters',                                        type=int,               default=1,                                                                                  help='Number of times to run the experiment, so we can obtain an average.')
    parser.add_argument('--prune',                                            action='store_true',                                                                                                help='Test with pruning.')
    parser.add_argument('--fine_tune',                                        action='store_true',                                                                                                help='Test with fine-tuning.')
    parser.add_argument('--confidence', type=float, default=0.999999, help='confidence value for recommending watermark size')
    parser.add_argument('--preserve_edges_between_subsets', action='store_true')

    parser.add_argument('--arch',                                             type=str,               default=config.node_classifier_kwargs['arch'],                                              help='GNN architecture (GAT, GCN, GraphConv, SAGE).')
    parser.add_argument('--activation',                                       type=str,               default=config.node_classifier_kwargs['activation'],                                        help='relu or elu.')
    parser.add_argument('--numLayers',                                          type=int,               default=config.node_classifier_kwargs['numLayers'],                                           help='Number of layers in GNN.')
    parser.add_argument('--hDim',                                             type=int,               default=config.node_classifier_kwargs['hDim'],                                              help='Number of hidden dimensions in GNN.')
    parser.add_argument('--dropout',                                          type=float,             default=config.node_classifier_kwargs['dropout'],                                           help='Dropout rate for classification training.')
    parser.add_argument('--dropout_subgraphs',                                type=float,             default=config.node_classifier_kwargs['dropout_subgraphs'],                                 help='Dropout rate for computing forward pass on watermarked subgraphs during training.')
    parser.add_argument('--skip_connections',                                 type=str2bool,          default=config.node_classifier_kwargs['skip_connections'],                                  help='Whether to include skip connections in GNN architecture: True (yes, true, t, y, 1) or False (no, false, f, n, 0).')
    parser.add_argument('--heads_1',                                          type=int,               default=config.node_classifier_kwargs['heads_1'],                                           help='Number of attention heads to use in first layer of GAT architecture.')
    parser.add_argument('--heads_2',                                          type=int,               default=config.node_classifier_kwargs['heads_2'],                                           help='Number of attention heads to use in intermediate layers of GAT architecture.')
    parser.add_argument('--use_batching',                                     type=str2bool,          default = config.node_classifier_kwargs['use_batching'],                                        help='Whether to use batching for forward pass (right now only for large data / Reddit2).')
    parser.add_argument('--batch_size',                                       type=int,               default = config.node_classifier_kwargs['batch_size'],                                        help='batch size for classification data -- right now batching only implemented for forward passes on classification data (not subgraphs)')

    parser.add_argument('--lr',                                           type=float,             default=config.optimization_kwargs['lr'],                                                   help='Learning rate.')
    parser.add_argument('--epochs',                                       type=int,               default=config.optimization_kwargs['epochs'],                                               help='Epochs.')
    parser.add_argument('--sacrifice_method',                             type=str,               default=config.optimization_kwargs['sacrifice_kwargs']['method'],                           help='If sacrificing some nodes from training, the method to use.')
    parser.add_argument('--clf_only',                                     type=str2bool,          default=config.optimization_kwargs['clf_only'],                                             help='Whether to train for classificaiton only (will skip watermarking).')
    parser.add_argument('--coefWmk',                                      type=float,             default=config.optimization_kwargs['coefWmk_kwargs']['coefWmk'],                            help='The coefficient on the watermarking loss term.')
    parser.add_argument('--use_pcgrad',                                   type=str2bool,          default=config.optimization_kwargs['use_pcgrad'],                                           help='Whether to use PCGrad to help mitigate conflicting gradients from multi-task learning.')

    parser.add_argument('--pGraphs',                                      type=float,             default=config.watermark_kwargs['pGraphs'],                                                 help='If using a multi-graph dataset, the proportion of graphs to watermark.')
    parser.add_argument('--watermark_type',                               type=str,               default=config.watermark_kwargs['watermark_type'],                                          help='Watermark type ("unimportant" indices vs "most_represented" indices).')

    parser.add_argument('--subgraph_regenerate',                              type=str2bool,          default=config.subgraph_kwargs['regenerate'],                                               help='Whether to regenerate subgraphs rather than load from local files (recommended if recent changes to code).')
    parser.add_argument('--subgraph_method',                                  type=str,               default=config.subgraph_kwargs['method'],                                                   help='Subgraph method (khop, random, rwr).')
    parser.add_argument('--subgraph_size_as_fraction',                                type=float,             default=config.subgraph_kwargs['subgraph_size_as_fraction'],                        help='Fraction of possible subgraph nodes comprising each watermarked subgraph.')
    parser.add_argument('--numSubgraphs',                                     type=int,               default=config.subgraph_kwargs['numSubgraphs'],                                             help='Number of subgraphs to watermark.')
    parser.add_argument('--subgraph_random_kwargs',                           type=dict,              default=config.subgraph_kwargs['random_kwargs'],                                            help='Empty dict -- no kwargs in current implementation.')

    
    parser.add_argument('--epsilon',                           type=float,             default=config.watermark_loss_kwargs['epsilon'],                                            help='Caps the influence of each nod feature index on the watermark loss. Smaller epislon = stricter cap.')

    parser.add_argument('--augment_separate_trainset_from_subgraphs',         type=str2bool,          default=config.augment_kwargs['separate_trainset_from_subgraphs'],                          help='Whether to augment regular training data separately from subgraphs used for watermarking.')
    parser.add_argument('--augment_p',                                        type=float,             default=config.augment_kwargs['p'],                                                         help='The proportion of data to augment.')
    parser.add_argument('--augment_ignore_subgraphs',                         type=str2bool,          default=config.augment_kwargs['ignore_subgraphs'],                                          help='If True, will not augment subgraphs used for watermarking.')
    parser.add_argument('--augment_nodeDrop_use',                             type=str2bool,          default=config.augment_kwargs['nodeDrop']['use'],                                           help='If True, will use nodeDrop augmentation.')
    parser.add_argument('--augment_nodeDrop_p',                               type=float,             default=config.augment_kwargs['nodeDrop']['p'],                                             help='If using nodeDrop augmentation, the probability of dropping a node.')
    parser.add_argument('--augment_nodeMixUp_use',                            type=str2bool,          default=config.augment_kwargs['nodeMixUp']['use'],                                          help='If True, will use nodeMixUp augmentation.')
    parser.add_argument('--augment_nodeMixUp_lambda',                         type=float,             default=config.augment_kwargs['nodeMixUp']['lambda'],                                       help='If using nodeMixUp augmentation, the relative ratio given to each node in the mixup (lambda, 1-lambda).')
    parser.add_argument('--augment_nodeFeatMask_use',                         type=str2bool,          default=config.augment_kwargs['nodeFeatMask']['use'],                                       help='If True, will use nodeFeatMask augmentation.')
    parser.add_argument('--augment_nodeFeatMask_p',                           type=float,             default=config.augment_kwargs['nodeFeatMask']['p'],                                         help='If using nodeFeatMask augmentation, the probability of masking node features.')
    parser.add_argument('--augment_edgeDrop_use',                             type=str2bool,          default=config.augment_kwargs['edgeDrop']['use'],                                           help='If True, will use edgeDrop augmentation.')
    parser.add_argument('--augment_edgeDrop_p',                               type=float,             default=config.augment_kwargs['edgeDrop']['p'],                                             help='If using edgeDrop augmentation, the probability of dropping an edge.')

    parser.add_argument('--watermark_random_backdoor', action='store_true')
    parser.add_argument('--watermark_random_backdoor_prob_edge', type=float, default=config.watermark_random_backdoor_prob_edge, help='**Hyperparamter for watermarking with backdoor trigger**: probability of edge connection in trigger graph.')
    parser.add_argument('--watermark_random_backdoor_proportion_ones',type=float,  default=config.watermark_random_backdoor_proportion_ones, help='**Hyperparamter for watermarking with backdoor trigger**: proportion of node features that are ones.')
    parser.add_argument('--watermark_random_backdoor_trigger_size_proportion', type=float, default=config.watermark_random_backdoor_trigger_size_proportion,help='**Hyperparamter for watermarking with backdoor trigger**: size of trigger graph as proportion of dataset.')
    parser.add_argument('--watermark_random_backdoor_trigger_alpha', type=float, default=config.watermark_random_backdoor_trigger_alpha, help='**Hyperparamter for watermarking with backdoor trigger**:relative weight for trigger loss')
    parser.add_argument('--watermark_random_backdoor_re_explain',action='store_true')


    parser.add_argument('--watermark_graphlime_backdoor', action='store_true')
    parser.add_argument('--watermark_graphlime_backdoor_target_label',type=int,default=config.watermark_graphlime_backdoor_target_label)
    parser.add_argument('--watermark_graphlime_backdoor_poison_rate',type=float,default=config.watermark_graphlime_backdoor_poison_rate)
    parser.add_argument('--watermark_graphlime_backdoor_size',type=float,default=config.watermark_graphlime_backdoor_size)

    args = parser.parse_args()

    config.preserve_edges_between_subsets = args.preserve_edges_between_subsets
    config.node_classifier_kwargs['arch']                                               = args.arch
    config.node_classifier_kwargs['activation']                                         = args.activation
    config.node_classifier_kwargs['numLayers']                                          = args.numLayers
    config.node_classifier_kwargs['hDim']                                               = args.hDim
    config.node_classifier_kwargs['dropout']                                            = args.dropout
    config.node_classifier_kwargs['dropout_subgraphs']                                  = args.dropout_subgraphs
    config.node_classifier_kwargs['skip_connections']                                   = args.skip_connections
    config.node_classifier_kwargs['heads_1']                                            = args.heads_1
    config.node_classifier_kwargs['heads_2']                                            = args.heads_2
    config.node_classifier_kwargs['use_batching']                                       = args.use_batching
    config.node_classifier_kwargs['batch_size']                                         = args.batch_size

    config.optimization_kwargs['lr']                                                    = args.lr
    config.optimization_kwargs['epochs']                                                = args.epochs
    config.optimization_kwargs['sacrifice_kwargs']['method']                            = args.sacrifice_method
    config.optimization_kwargs['clf_only']                                              = args.clf_only
    config.optimization_kwargs['coefWmk_kwargs']['coefWmk']                             = args.coefWmk
    config.optimization_kwargs['use_pcgrad']                                            = args.use_pcgrad

    config.watermark_kwargs['pGraphs']                                                  = args.pGraphs
    config.watermark_kwargs['watermark_type']                                           = args.watermark_type

    config.subgraph_kwargs['regenerate']                                                = args.subgraph_regenerate
    config.subgraph_kwargs['numSubgraphs']                                              = args.numSubgraphs
    config.subgraph_kwargs['method']                                                    = args.subgraph_method
    config.subgraph_kwargs['subgraph_size_as_fraction']                                 = args.subgraph_size_as_fraction
    config.subgraph_kwargs['random_kwargs']                                             = args.subgraph_random_kwargs


    config.watermark_loss_kwargs['epsilon']                                             = args.epsilon

    config.augment_kwargs['separate_trainset_from_subgraphs']                           = args.augment_separate_trainset_from_subgraphs
    config.augment_kwargs['p']                                                          = args.augment_p
    config.augment_kwargs['ignore_subgraphs']                                           = args.augment_ignore_subgraphs
    config.augment_kwargs['nodeDrop']['use']                                            = args.augment_nodeDrop_use
    config.augment_kwargs['nodeDrop']['p']                                              = args.augment_nodeDrop_p
    config.augment_kwargs['nodeMixUp']['use']                                           = args.augment_nodeMixUp_use
    config.augment_kwargs['nodeMixUp']['lambda']                                        = args.augment_nodeMixUp_lambda
    config.augment_kwargs['nodeFeatMask']['use']                                        = args.augment_nodeFeatMask_use
    config.augment_kwargs['nodeFeatMask']['p']                                          = args.augment_nodeFeatMask_p
    config.augment_kwargs['edgeDrop']['use']                                            = args.augment_edgeDrop_use
    config.augment_kwargs['edgeDrop']['p']                                              = args.augment_edgeDrop_p

    config.watermark_random_backdoor = args.watermark_random_backdoor
    config.watermark_random_backdoor_prob_edge = args.watermark_random_backdoor_prob_edge
    config.watermark_random_backdoor_proportion_ones = args.watermark_random_backdoor_proportion_ones
    config.watermark_random_backdoor_trigger_size_proportion = args.watermark_random_backdoor_trigger_size_proportion
    config.watermark_random_backdoor_trigger_alpha = args.watermark_random_backdoor_trigger_alpha

    config.watermark_graphlime_backdoor=args.watermark_graphlime_backdoor
    config.watermark_graphlime_backdoor_target_label = args.watermark_graphlime_backdoor_target_label
    config.watermark_graphlime_backdoor_poison_rate = args.watermark_graphlime_backdoor_poison_rate
    config.watermark_graphlime_backdoor_size = args.watermark_graphlime_backdoor_size
    config.watermark_random_backdoor_re_explain=args.watermark_random_backdoor_re_explain

    if args.watermark_random_backdoor==False and args.clf_only==False and args.watermark_graphlime_backdoor==False:
        args.using_our_method=True
        config.using_our_method = True
    else:
        args.using_our_method=False
        config.using_our_method = False

    config.seed = args.seed

    n_features = data.x.shape[1]

    if args.clf_only==False:
        c = config.subgraph_kwargs['numSubgraphs']
        mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
        c_LB=args.confidence
        c_t=args.confidence
        recommended_watermark_length = find_min_n_uncertain(n_features, mu_natural, sigma_natural, c_LB, c_t, test_effective=True, verbose=True)
        recommended_percent = 100*recommended_watermark_length/n_features
        config.watermark_kwargs['percent_of_features_to_watermark']=recommended_percent
        title_ = f'Separate Forward Passes -- {config.watermark_kwargs["watermark_type"]} feature indices'
        title = f'{title_}.\n{config.subgraph_kwargs["numSubgraphs"]} subgraphs.\nWatermarking {config.watermark_kwargs["percent_of_features_to_watermark"]}% of node features'
        z_t=norm.ppf(args.confidence)
        target_number_matches = np.ceil(min(mu_natural +z_t*sigma_natural,data.x.shape[1]))

    data_original = copy.deepcopy(data)


    for _ in range(args.num_iters):
        data = copy.deepcopy(data_original)

        if dataset_name=='PubMed':# 
            if args.clf_only==True or config.watermark_random_backdoor==True or config.watermark_graphlime_backdoor==True:
                config.augment_kwargs['nodeMixUp']['lambda']=0.6#>0.1
                config.augment_kwargs['nodeDrop']['p']=0.1#>0.1
                config.augment_kwargs['nodeFeatMask']['p']=0.1#>0.1
                config.augment_kwargs['edgeDrop']['p']=0.1#>0.1

        Trainer_ = Trainer(data, dataset_name, None if args.clf_only else target_number_matches)


        results = Trainer_.train(debug_multiple_subgraphs=False, save=True, print_every=1)
        results_folder_name = get_results_folder_name(dataset_name)

        if config.using_our_method==False:
            results_folder_name = get_results_folder_name(dataset_name)
            node_classifier, history = results
            primary_loss_curve = history['losses_primary']
            train_acc = history['train_accs'][-1]
            val_acc = history['val_accs'][-1]
            test_acc = history['test_accs'][-1]
            epoch = config.optimization_kwargs['epochs']-1
            loss_prim = primary_loss_curve[-1]

            if args.clf_only==True:
                final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, clf_loss = {loss_prim:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}'
                results_file_name = 'results_clf_only.txt'
            elif config.watermark_random_backdoor==True:
                trigger_acc = history['trigger_accs'][-1]
                final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, clf_loss = {loss_prim:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}, trigger acc = {trigger_acc:.3f}'
                results_file_name = 'results_watermark_random_backdoor.txt'
            elif config.watermark_graphlime_backdoor==True:
                graphlime_backdoor_acc = history['graphlime_backdoor_accs'][-1]
                final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, clf_loss = {loss_prim:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}, backdoor nodes acc = {graphlime_backdoor_acc:.3f}'
                results_file_name = 'results_watermark_graphlime_backdoor.txt'
            model_config_results_filepath = os.path.join(results_folder_name,results_file_name)

            with open(model_config_results_filepath,'a') as f:
                f.write(final_performance + '\n')
            f.close()

            print(final_performance)
        if config.using_our_method==True and args.clf_only==False:
            node_classifier, history, subgraph_dict, all_feature_importances, all_watermark_indices  = results

            primary_loss_curve, watermark_loss_curve, final_betas, watermarks, percent_matches, percent_match_mean, percent_match_std, primary_acc_curve, watermark_acc_curve, train_acc, val_acc, test_acc, match_counts_with_zeros, match_counts_without_zeros,match_count_confidence_with_zeros,match_count_confidence_without_zeros = get_performance_trends(history, subgraph_dict, config.optimization_kwargs)

            epoch = config.optimization_kwargs['epochs']-1
            loss_prim = primary_loss_curve[-1]
            loss_watermark = watermark_loss_curve[-1]
            percent_match=percent_matches[-1]
            train_acc=train_acc

            final_performance = f'Seed {seed}\nEpoch: {epoch:3d}, L (clf/wmk) = {loss_prim:.3f}/{loss_watermark:.3f}, acc (trn/val/test)= {train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f}, #_match_WMK w/wout 0s = {match_counts_with_zeros}/{match_counts_without_zeros}, conf w/wout 0s = {match_count_confidence_with_zeros:.3f}/{match_count_confidence_without_zeros:.3f}'
            results_folder_name = get_results_folder_name(dataset_name)
            results_file_name = 'results.txt' if args.fine_tune==False else 'results_fine_tune.txt'
            model_config_results_filepath = os.path.join(results_folder_name,results_file_name)


            mu_natural, sigma_natural = get_natural_match_distribution(n_features, c)
            if os.path.exists(model_config_results_filepath)==False:
                with open(model_config_results_filepath,'w') as f:
                    f.write(f'Natural match distribution: mu={mu_natural:.3f}, sigma={sigma_natural:.3f}\n')
                f.close()        
            with open(model_config_results_filepath,'a') as f:
                f.write(final_performance)
            f.close()

            plot_name = dataset_name
            save_fig=True if args.fine_tune==False else False
            print(final_performance)
        

        del Trainer_
        del results
        del node_classifier
        del history
        
        seed += 1
        set_seed(seed)
        config.seed=seed




