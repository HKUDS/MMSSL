import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="")

    ### imdb ########################################################################################
    ##no tune
    parser.add_argument('--dataset', type=str, default='imdb')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  #3407
    parser.add_argument('--model', type=str, default="graphsage") #gcn, gat, graphsage, sgc
    parser.add_argument('--feat_norm', type=int, default=-1)
    parser.add_argument('--adj_norm_order', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)  
    parser.add_argument('--cfg_scale', type=float, default=0.6)  
    parser.add_argument('--edge_reconstr_scale', type=float, default=1)
    parser.add_argument('--ema', type=bool, default=True)  # we use label 622
    parser.add_argument('--ema_scale', type=float, default=0.999)
    parser.add_argument('--edge_loss_batch', type=int, default=4096)
    ##common
    parser.add_argument('--lr', type=float, default=0.1)  #0.1, 0.07
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument("--lr_decoder", type=float, default=0.1, help="")
    parser.add_argument("--weight_decay_decoder", type=float, default=5e-4, help="")
    parser.add_argument('--dropout', type=float, default=0)
    ##adaptive diffusion
    parser.add_argument('--com_feat_dim', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--diffusion_type', type=str, default='dhs_ppr') # raw, dhs, ppr, hk, dhs_ppr, ppr_dhs
    parser.add_argument('--total_step', type=int, default=300)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0)  #0, 0.2, 0.4, 0.6, 0.8, 1
    parser.add_argument('--if_ori_g', type=bool, default=False) 
    ##cfg&prune_loss
    parser.add_argument('--cfg', type=bool, default=True) 
    parser.add_argument('--cfg_drop_rate', type=float, default=0.8) #0.8
    # parser.add_argument("--task_rele_l_drop_rate", type=float, default=0.9, help="")
    parser.add_argument("--tip_rate_edge", type=float, default=0.9, help="")  #1:6803,6791 ! ^ ~
    parser.add_argument("--tip_rate_feat", type=float, default=0.5, help="")  #0.2 ! ^ ~    
    ##reverse
    parser.add_argument('--pos_samp_num', type=int, default=20) #1, 2, 3, 7, 11, 12
    parser.add_argument('--neg_samp_num', type=int, default=7)
    parser.add_argument('--l_edge_rate', type=float, default=0.5) #0, 0.1, 0.3, [0.5], 0.7, 0.9, 1.3, 1.5, 1.7, 1.9, 2.5
    parser.add_argument('--l_feat_rate', type=float, default=1) #sce[0, [1], 3, 5, 7, 9, 11, 13, 15, 17, 19, 0.5, 1.5]  mse[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    parser.add_argument('--subgraph_by_weight', type=bool, default=False)  # we use label 622
    # parser.add_argument('--random_walk_len', type=int, default=5)
    parser.add_argument('--sample_type', type=str, default='neighbor') # node2vec_radom_walk, random_walk
    parser.add_argument('--feat_loss_type', type=str, default='sce') # mse, sce
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
    parser.add_argument('--feature_mask_rate', type=float, default=0.0)

    parser.add_argument('--point', type=str, default='l_edge_0__1')
    ### imdb ########################################################################################

# ### acm ########################################################################################
#     ##no tune
#     parser.add_argument('--dataset', type=str, default='acm')
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=0)  #3407
#     parser.add_argument('--model', type=str, default="graphsage") #gcn, gat, graphsage, sgc
#     parser.add_argument('--feat_norm', type=int, default=-1)
#     parser.add_argument('--adj_norm_order', type=int, default=1)
#     parser.add_argument('--alpha', type=float, default=0)
#     parser.add_argument('--cfg_scale', type=float, default=0.6)
#     parser.add_argument('--edge_reconstr_scale', type=float, default=1)
#     parser.add_argument('--ema', type=bool, default=True)  # we use label 622
#     parser.add_argument('--ema_scale', type=float, default=0.999)
#     parser.add_argument('--edge_loss_batch', type=int, default=4096)
#     parser.add_argument('--feature_mask_rate', type=float, default=0.11)
#     parser.add_argument('--dropout', type=float, default=0)
#     ##common
#     parser.add_argument('--lr', type=float, default=0.04)  #0.04:maf1/0.9337,mif1/0.9372  0.05:test_f1/0.9348, test_mif1/0.9383  0.042: maf1/0.9364, mif1/0.9394 
#     parser.add_argument('--weight_decay', type=float, default=0.00013)  #0.00007:test_f1/0.9351, test_mif1/0.9383, 0.00008:'test_f1': '0.9373', 'test_mif1': '0.9405' 0.00012:test_f1/0.9373, test_mif1/0.9405 0.00013:test_f1/0.9361, test_mif1/0.9394
#     parser.add_argument('--epochs', type=int, default=200)  #150, 250, 350
#     parser.add_argument('--early_stop', type=int, default=40)
#     parser.add_argument("--lr_decoder", type=float, default=0.011, help="")
#     parser.add_argument("--weight_decay_decoder", type=float, default=5e-4, help="")
#     ##adaptive diffusion
#     parser.add_argument('--com_feat_dim', type=int, default=64) #!
#     parser.add_argument('--emb_dim', type=int, default=256)
#     parser.add_argument('--diffusion_type', type=str, default='dhs_ppr') # raw, dhs, ppr, hk, dhs
#     parser.add_argument('--total_step', type=int, default=1000)
#     parser.add_argument('--num_head', type=int, default=2) #!
#     parser.add_argument('--threshold', type=float, default=0.6)  #!
#     parser.add_argument('--if_ori_g', type=bool, default=True) # ^
#     ##cfg&prune_loss
#     parser.add_argument('--cfg', type=bool, default=True)  #! ~
#     parser.add_argument('--cfg_drop_rate', type=float, default=0.9) # ^
#     parser.add_argument("--tip_rate_edge", type=float, default=0.4, help="")  # ! ^ ~
#     parser.add_argument("--tip_rate_feat", type=float, default=0, help="")  #0.2 ! ^ ~
#     ##reverse     
#     parser.add_argument('--pos_samp_num', type=int, default=11) # ^  1, 2, 3, 4, 5, 6, 7, 8
#     parser.add_argument('--neg_samp_num', type=int, default=5)  # ^
#     parser.add_argument('--l_edge_rate', type=float, default=0.04) # ! 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, [0.9], 1, 1.3, 1.5, 1.7, 1.9
#     parser.add_argument('--l_feat_rate', type=float, default=17) # ! sce[0, 0.5, 1, 3, 5, 7, 9, 11, 13, 15, [17], 19, 25, 0.5, 1.5]
#     parser.add_argument('--subgraph_by_weight', type=bool, default=True)  # ^
#     # parser.add_argument('--random_walk_len', type=int, default=5)
#     # parser.add_argument('--random_walk_len', type=int, default=5)
#     parser.add_argument('--sample_type', type=str, default='neighbor') # ^ node2vec_radom_walk, random_walk
#     parser.add_argument('--feat_loss_type', type=str, default='sce') # ^ mse, sce
#     parser.add_argument("--alpha_l", type=float, default=2, help="!`pow`inddex for `sce` loss")  #$

#     parser.add_argument('--point', type=str, default='l_edge_loss__1_weight')
# ### acm ########################################################################################


#     ### dblp ########################################################################################
#     ##no tune
#     parser.add_argument('--dataset', type=str, default='dblp')
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=0)  #3407
#     parser.add_argument('--model', type=str, default="graphsage") #gcn, gat, graphsage, sgc
# #    parser.add_argument('--feat_norm', type=int, default=-1)
#     parser.add_argument('--adj_norm_order', type=int, default=1)
#     parser.add_argument('--alpha', type=float, default=1)
#     parser.add_argument('--cfg_scale', type=float, default=0.6)
#     parser.add_argument('--edge_reconstr_scale', type=float, default=0.1)
#     parser.add_argument('--ema', type=bool, default=True)  # we use label 622
#     parser.add_argument('--ema_scale', type=float, default=0.999)
#     parser.add_argument('--edge_loss_batch', type=int, default=4096)
#     ##common
#     parser.add_argument('--lr', type=float, default=0.06)
#     parser.add_argument('--weight_decay', type=float, default=0.0001)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--early_stop', type=int, default=30)
#     parser.add_argument("--lr_decoder", type=float, default=0.1, help="")
#     parser.add_argument("--weight_decay_decoder", type=float, default=5e-4, help="")
#     parser.add_argument('--dropout', type=float, default=0.2) 

#     ##adaptive diffusion
#     parser.add_argument('--com_feat_dim', type=int, default=128)
#     parser.add_argument('--emb_dim', type=int, default=64)
#     parser.add_argument('--diffusion_type', type=str, default='dhs_ppr') # raw, dhs, ppr, hk, dhs_ppr
#     parser.add_argument('--total_step', type=int, default=400)

#     parser.add_argument('--num_head', type=int, default=2)
#     parser.add_argument('--threshold', type=float, default=0)  #[0, 0.2, [0.4], 0.6, 0.8, 1]
#     parser.add_argument('--if_ori_g', type=bool, default=True) 
#     ##cfg&prune_loss
#     parser.add_argument('--cfg', type=bool, default=True)  # we use label 622
#     parser.add_argument('--cfg_drop_rate', type=float, default=0.8) #0.8
#     #parser.add_argument("--task_rele_l_drop_rate", type=float, default=0.9, help="")
#     parser.add_argument("--tip_rate_edge", type=float, default=0.9, help="")  # ! ^ ~
#     parser.add_argument("--tip_rate_feat", type=float, default=0, help="")  #0.2 ! ^ ~
#     ##reverse
#     parser.add_argument('--pos_samp_num', type=int, default=2)  #1,2,3,4,5,6,7,8,9,10,11,12
#     parser.add_argument('--neg_samp_num', type=int, default=5)
#     parser.add_argument('--l_edge_rate', type=float, default=0) #0, 0.1, [0.3], 0.5, 0.7, 0.9 ,1.3, 1.5 ,1.7, 2, 2.5
#     parser.add_argument('--l_feat_rate', type=float, default=7)  ##sce[0, 0.5, 1, 3, 5, [7], 9, 11, 13, 15, 17, 19, 0.5, 1.5]
#     parser.add_argument('--subgraph_by_weight', type=bool, default=True)  # we use label 622
#     # parser.add_argument('--random_walk_len', type=int, default=5)
#     parser.add_argument('--sample_type', type=str, default='neighbor') # node2vec_radom_walk, random_walk
#     parser.add_argument('--feat_loss_type', type=str, default='sce') # mse, sce
#     parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
#     parser.add_argument('--feature_mask_rate', type=float, default=0.11)


#     parser.add_argument('--point', type=str, default='neighbor_base')
#     ### dblp ########################################################################################




    # ### yelp ########################################################################################
    # parser.add_argument('--dataset', type=str, default='yelp')
    # parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--seed', type=int, default=0)  #3407

    # parser.add_argument('--lr', type=float, default=0.03)
    # parser.add_argument('--weight_decay', type=float, default=0.0001)
    # parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--com_feat_dim', type=int, default=16)
    # parser.add_argument('--emb_dim', type=int, default=64)
    # parser.add_argument('--feat_norm', type=int, default=-1)
    # parser.add_argument('--adj_norm_order', type=int, default=1)
    # parser.add_argument('--early_stop', type=int, default=40)
    # parser.add_argument('--num_head', type=int, default=2)
    # parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--dropout', type=float, default=0)

    # parser.add_argument('--diffusion_type', type=str, default='ppr') # raw, dhs, ppr, hk, dhs
    # parser.add_argument('--cfg', type=bool, default=True) 
    # parser.add_argument('--cfg_scale', type=float, default=0.6)
    # parser.add_argument('--cfg_drop_rate', type=float, default=0.8) #0.8
    # parser.add_argument('--edge_reconstr_scale', type=float, default=0.1)
    # parser.add_argument('--pos_samp_num', type=int, default=1)
    # parser.add_argument('--neg_samp_num', type=int, default=5)
    # parser.add_argument('--l_edge_loss_rate', type=float, default=0.0001)
    # parser.add_argument('--ema', type=bool, default=True)  # we use label 622
    # parser.add_argument('--ema_scale', type=float, default=0.999)
    # parser.add_argument('--edge_loss_batch', type=int, default=4096)


    # parser.add_argument('--point', type=str, default='base_dhs')
    # ### yelp ########################################################################################


#@@GCN, GAT, GraphSage@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
    # ### imdb ########################################################################################
    # parser.add_argument('--dataset', type=str, default='imdb')
    # parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--seed', type=int, default=0)  #3407

    # parser.add_argument('--lr', type=float, default=0.03)
    # parser.add_argument('--weight_decay', type=float, default=0.0001)
    # parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--com_feat_dim', type=int, default=16)
    # parser.add_argument('--emb_dim', type=int, default=64)
    # parser.add_argument('--feat_norm', type=int, default=-1)
    # parser.add_argument('--adj_norm_order', type=int, default=1)
    # parser.add_argument('--early_stop', type=int, default=80)
    # parser.add_argument('--num_head', type=int, default=2)
    # parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--dropout', type=float, default=0)

    # parser.add_argument('--model', type=str, default="gcn") #gcn, gat, graphsage, sgc

    # # parser.add_argument('--point', type=str, default='l_edge_0__1')
    # ### imdb ########################################################################################



# ### acm ########################################################################################
#     parser.add_argument('--dataset', type=str, default='acm')
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=0)  #3407

#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--weight_decay', type=float, default=0.0001)
#     parser.add_argument('--epochs', type=int, default=500)
#     parser.add_argument('--com_feat_dim', type=int, default=16)
#     parser.add_argument('--emb_dim', type=int, default=64)
#     parser.add_argument('--feat_norm', type=int, default=-1)
#     parser.add_argument('--adj_norm_order', type=int, default=1)
#     parser.add_argument('--early_stop', type=int, default=500)
#     parser.add_argument('--num_head', type=int, default=2)
#     parser.add_argument('--alpha', type=float, default=1)
#     parser.add_argument('--dropout', type=float, default=0)

#     parser.add_argument('--model', type=str, default="gcn") #gcn, gat, graphsage, sgc

#     # parser.add_argument('--point', type=str, default='l_edge_loss__1_weight')
# ### acm ########################################################################################



    # ### dblp ########################################################################################
    # parser.add_argument('--dataset', type=str, default='dblp')
    # parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--seed', type=int, default=0)  #3407

    # parser.add_argument('--lr', type=float, default=0.005)
    # parser.add_argument('--weight_decay', type=float, default=0.0001)
    # parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--com_feat_dim', type=int, default=16)
    # parser.add_argument('--emb_dim', type=int, default=64)
    # parser.add_argument('--feat_norm', type=int, default=-1)
    # parser.add_argument('--adj_norm_order', type=int, default=1)
    # parser.add_argument('--early_stop', type=int, default=80)
    # parser.add_argument('--num_head', type=int, default=2)
    # parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--dropout', type=float, default=0)

    # parser.add_argument('--model', type=str, default="gin") #gcn, gat, graphsage, sgc, gin
    # parser.add_argument('--lp_batch_size', type=int, default=1024, help='link prediction batch size.')
    # parser.add_argument('--Ks', default='[10, 20, 50]', help='K value of ndcg/recall @ k')

    # # parser.add_argument('--point', type=str, default='neighbor_base')
    # ### dblp ########################################################################################


    return parser.parse_args()
