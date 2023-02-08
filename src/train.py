import os
import sys
import optuna
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)
from early_stopper import *
from hin_loader import HIN
from evaluation import *
import util_funcs as uf
from util_funcs import *
from config import DHSConfig
from DHS import DHS, EMA, Decoder

import parser
import warnings
import time
import torch
import argparse
import torch.nn.functional as F
import pickle
import dgl

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC


warnings.filterwarnings('ignore')
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]



args = parser.parse_args()

BEST_RESULT = 0
BEST_VALUE = 0

def train_dhs(args, gpu_id=0, log_on=True):
    # uf.seed_init(args.seed)
    uf.shell_init(gpu_id=gpu_id)
    cf = DHSConfig(args.dataset)

    # ! Modify config
    cf.update(args.__dict__)
    cf.dev = torch.device("cuda:0" if gpu_id >= 0 else "cpu")

    # ! Load Graph
    g = HIN(cf.dataset).load_mp_embedding(cf)
    print(f'Dataset: {cf.dataset}, {g.t_info}')
    # features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y, dgl_pos, dgl_neg = g.to_torch(cf)
    features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y, ppr_adj = g.to_torch(cf)


    # ! Train Init
    if not log_on: uf.block_logs()
    print(f'{cf}\nStart training..')
    cla_loss = torch.nn.NLLLoss()
    model = DHS(cf, g, args)
    model.to(cf.dev)
    decoder = Decoder(features, cf, g).cuda()

    """
    EMA part1:
    """
    if args.ema:
        #init
        ema = EMA(args.ema_scale)  #0.995, 0.999
        #register
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    # optimizer = torch.optim.Adam(
    #      [{'params':model.parameters(), 'initial_lr':args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.AdamW(decoder.parameters(), lr = args.lr_decoder, weight_decay=args.weight_decay_decoder)
    # schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=10, step_size_down=None, mode='triangular2', cycle_momentum=False, gamma=0.9)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)
    #init feat score
    feat_score = torch.rand(features.shape[0])

    """
    visualization
    """
    line_loss, line_train_f1, line_test_f1, line_val_f1, line_train_mif1, line_test_mif1, line_val_mif1 = [], [], [], [], [], [], []

    dur = []
    w_list = []
    for epoch in range(cf.epochs):
        #forward process: features, adj, mp_emb
        """
        1) get raw data
        2) input time step t, output noised data at time step t. 
                              output noised t-1 as ground truth.
        3)
        """

        #reverse process: [ground truth]

        # ! Train
        t0 = time.time()
        model.train()
        # args.cfg = False
        # if np.random.random() < args.cfg_drop_rate:
        #     args.cfg = True
        # else:
        #     args.cfg = False

        if np.random.random() < args.cfg_drop_rate:
            args.cfg = False
        # else:
        #     args.cfg = False

        logits, adj_new, x, hid_emb, t_1_noise_adj, t_1_index = model(features, adj, mp_emb, train_x, train_y, ppr_adj, feat_score, decoder)
        """
        edge reconstruction
        """
        # ## Version1: HGMAE #############################################
        # # Step1: A' = H.H^T
        # # Step2: L = (1/|A|)sum( 1 - A.A'/|A||A'| )
        # # Step3: (learnble)  s = MLP(H) \alpha = softmax(s)
        # # Step4: L_{MER} =  sum(\alpha . L )
        # # adj_new
        # x_norm = F.normalize(hid_emb)
        # A_pre = F.sigmoid(torch.matmul(hid_emb, hid_emb.T)) #logits, x, hid_emb
        # l_edge_loss =  (1 - (torch.multiply(adj_new, A_pre))).mean()  # (1)mat need normalize  (2)
        # print(f"#######################################l_edge_loss: {l_edge_loss}")

        ### Version2: MaskGAE--binary cross-emtropy #############################################
        # Step1: sample 
        # Step2: CE

        ### Version3: myself #############################################
        # Step1: sample 
        # Step2: loss

        # #decoder
        # pos_row, pos_col, neg_row, neg_col = uf.dgl_sample(dgl_pos, dgl_neg, args.pos_samp_num, args.neg_samp_num, torch.range(0, adj_new.shape[0]-1, dtype=torch.int64).cuda())  #, cf.node_list[0], cf.relation_type_list) #
        # l_gae_edge = 0
        # # for l_i, l_val in enumerate(cf.relation_type_list):
        # #     pos_row = pos_row_list[l_i]
        # #     pos_col = pos_col_list[l_i]
        # #     neg_row = neg_row_list[l_i]
        # #     neg_col = neg_col_list[l_i]

        # pos_row_embed = hid_emb[pos_row]  
        # pos_col_embed = hid_emb[pos_col]
        # neg_row_embed = hid_emb[neg_row]
        # neg_col_embed = hid_emb[neg_col]

        # pred_i, pred_j = uf.innerProduct(pos_row_embed, pos_col_embed, neg_row_embed, neg_col_embed)  
        # l_gae_edge += (- ((pred_i.sum(dim=1)/(args.pos_samp_num**1)).sigmoid()+1e-8).log() - ((- pred_j.sum(dim=1)/(args.neg_samp_num**1)).sigmoid()+1e-8).log()).sum()

        ### Version4: dgl_sample#############################################
        # adj_re = torch.multiply(adj, adj_new)
        adj_re = t_1_noise_adj
        # adj_re = adj_new
        x_index, y_index = torch.where(adj_re!=0)[0], torch.where(adj_re!=0)[1] 
        dgl_g = dgl.graph((x_index, y_index)).to('cuda:0')
        dgl_g.edata['weight'] = adj_re[x_index, y_index]
        # node_id = torch.arange(adj_re.shape[0]).cuda()
        # node_id = torch.arange( cf.id_target_end ).cuda()
        node_id = torch.arange( cf.id_target_end ).cuda()

        # # #---------------------------------------------------------------------------------------------------------------
        # positive_x, positive_y = g.edges()[0], g.edges()[1] 
        # # positive_x, positive_y = positive_x.repeat(1, args.neg_samp_num).reshape(-1), positive_y.repeat(1, args.neg_samp_num).reshape(-1) 
        # neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.neg_samp_num)
        # negative_x, negative_y = neg_sampler(g, torch.arange(g.edges()[0].shape[0]).cuda())
        # # batch loss-----------------------------------------------------------------------------------------------------
        # num_batches = (g.edges()[0].shape[0] - 1) // args.edge_loss_batch + 1
        # indices = torch.arange(0, g.edges()[0].shape[0]).cuda()
        # l_edge_loss = 0
        # for i in range(num_batches):
        #     tmp_i = indices[i * args.edge_loss_batch:(i + 1) * args.edge_loss_batch]
        #     pos_row_embed = hid_emb[positive_x[tmp_i]]  
        #     pos_col_embed = hid_emb[positive_y[tmp_i]]
        #     neg_row_embed = hid_emb[negative_x[tmp_i]]
        #     neg_col_embed = hid_emb[negative_y[tmp_i]]

        #     pred_i, pred_j = uf.innerProduct(pos_row_embed, pos_col_embed, neg_row_embed, neg_col_embed)  
            # l_edge_loss += - ((pred_i.view(-1) - pred_j.view(-1)).sigmoid()+1e-8).log().mean()  
        # # #---------------------------------------------------------------------------------------------------------------

        # random_walk:
        if args.sample_type=='random_walk':
            path, _ = dgl.sampling.random_walk(dgl_g, node_id, length=args.pos_samp_num)      
            node_id_repeat = node_id.unsqueeze(1).repeat(1, args.pos_samp_num+1)
            path = path.view(-1)
            node_id_repeat = node_id_repeat.view(-1)
            pos_row_embed = hid_emb[node_id_repeat]
            pos_col_embed = hid_emb[path]
            pred_i = uf.innerProduct(pos_row_embed, pos_col_embed)  
            l_edge = - (pred_i.view(-1) .sigmoid()+1e-8).log().sum()  
        elif args.sample_type=='node2vec_radom_walk':
            #node2vec_radom_walk:
            dgl_g = dgl_g.to('cpu')
            node_id = node_id.cpu()
            path = dgl.sampling.node2vec_random_walk(dgl_g, node_id, p=0.4, q=1, walk_length=args.pos_samp_num)
            node_id_repeat = node_id.unsqueeze(1).repeat(1, args.pos_samp_num+1)
            path = path.view(-1)
            node_id_repeat = node_id_repeat.view(-1)
            pos_row_embed = hid_emb[node_id_repeat]
            pos_col_embed = hid_emb[path]
            pred_i = uf.innerProduct(pos_row_embed, pos_col_embed)  
            l_edge = - (pred_i.view(-1) .sigmoid()+1e-8).log().sum() 
        elif args.sample_type=='neighbor':
            # neighbor:
            if args.subgraph_by_weight:
                sub_graph = dgl.sampling.sample_neighbors(dgl_g, node_id, fanout=args.pos_samp_num, prob='weight')
            else:
                sub_graph = dgl.sampling.sample_neighbors(dgl_g, node_id, fanout=args.pos_samp_num)
            x_index, y_index = sub_graph.edges()
            # #neighbor_biased:
            # sub_graph = dgl.sampling.sample_neighbors_biased(g, node_id, fanout=5, )
            pos_row_embed = hid_emb[x_index]
            pos_col_embed = hid_emb[y_index]
            pred_i = uf.innerProduct(pos_row_embed, pos_col_embed)  
            # l_edge_loss = - (pred_i.view(-1) .sigmoid()+1e-8).log().sum() 

            # l_edge = - (pred_i.view(-1)+1e-8).log().mean() 
            l_edge = - (pred_i.view(-1)+1e-8).log()
            if args.tip_rate_edge!=0:
                l_edge = uf.loss_function(l_edge, args.tip_rate_edge)   
            else:
                l_edge = l_edge.mean() 
            # print(f"#######################################l_edge_loss: {l_edge_loss}")
        # elif args.sample_type=='neighbor':
        #     # neighbor:
        #     sub_graph = dgl.sampling.sample_neighbors(g, node_id, fanout=args.pos_samp_num)
        #     x_index, y_index = sub_graph.edges()
        #     # #neighbor_biased:
        #     # sub_graph = dgl.sampling.sample_neighbors_biased(g, node_id, fanout=5, )
        #     pos_row_embed = hid_emb[x_index]
        #     pos_col_embed = hid_emb[y_index]
        #     pred_i = uf.innerProduct(pos_row_embed, pos_col_embed)  
        #     l_edge_loss = - (pred_i.view(-1) .sigmoid()+1e-8).log().sum() 
        #     print(f"#######################################l_edge_loss: {l_edge_loss}")
        #     # #select_topk:
        #     # g = g.to('cpu')
        #     # node_id = node_id.cpu()
        #     # sub_graph = dgl.sampling.select_topk(g, k=5, weight='weight', nodes=node_id)


        ### Version5: directly#############################################
        """
        attribute restoration
        """
        #decode
        # for et_id, t_value in enumerate(mask_nodes_dict):
        #     feat_learned[mask_nodes_dict[t_value]] = 0.0
        reconstru_feat = decoder(hid_emb, adj_new)  #logits, adj_new, x, hid_emb
        feat_score = uf.get_feat_score(reconstru_feat, features)

        # reconstru_feat = decoder(feat_learned)
        if args.feat_loss_type=='mse':
            # feature_loss = sce_criterion(reconstru_feat, features, mask_nodes_dict, alpha=args.alpha_l)
            l_feat = uf.mse_criterion(reconstru_feat, features, alpha=args.alpha_l)
        elif args.feat_loss_type=='sce':
            l_feat = uf.sce_criterion(reconstru_feat, features, alpha=args.alpha_l, tip_rate=args.tip_rate_feat)

        train_f1, train_mif1 = eval_logits(logits, train_x, train_y)
        w_list.append(uf.print_weights(model))
        l_pred = cla_loss(logits[train_x], train_y)
        loss = l_pred + args.l_edge_rate*l_edge + args.l_feat_rate*l_feat
        print(f"l_pred:{l_pred}   edge_reconstr_loss:{args.l_edge_rate*l_edge}   l_feat:{args.l_feat_rate*l_feat}  ")
        line_loss.append(loss.item())
        optimizer.zero_grad()
        optimizer_decoder.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()
        optimizer_decoder.step()
        # schedule.step()
        torch.cuda.empty_cache()

        """
        EMA part2:
        """
        if args.ema:
            # update
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.update(name, param.data) 

        # ! Valid
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            logits, x, hid_emb = model.GCN(features, adj_new)
            train_f1, train_mif1 = eval_logits(logits, train_x, train_y)
            val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
            test_f1, test_mif1 = eval_logits(logits, test_x, test_y)

            test_f1_, test_mif1_, val_f1_, val_mif1_ = eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper)

        dur.append(time.time() - t0)
        uf.print_train_log(epoch, dur, loss, train_f1, val_f1)
        line_train_f1.append(train_f1.item())
        # line_test_f1.append(test_f1.item())
        line_test_f1.append(test_f1_.item())
        line_val_f1.append(val_f1.item())
        line_train_mif1.append(train_mif1.item())
        # line_test_mif1.append(test_mif1.item())
        line_test_mif1.append(test_mif1_.item())
        line_val_mif1.append(val_mif1.item())
        print(f"###test_f1: {test_f1}, ###test_mif1: {test_mif1}")

        # line_var_ndcg.append(ret['ndcg'][1])

        DHS_result = {
            "loss": line_loss,
            "train_f1": line_train_f1,
            "test_f1": line_test_f1,
            "var_f1": line_val_f1,
            "train_mif1": line_train_mif1,
            "test_mif1": line_test_mif1,
            "var_mif1": line_val_mif1,
            # "var_ndcg": line_var_ndcg,
        }
        pickle.dump(DHS_result, open('/home/ww/FILE_from_ubuntu18/Code/work7/plot/'+args.dataset+'/'+args.point,'wb'))

        if cf.early_stop > 0:
            if stopper.step(val_f1, model, epoch):
                print(f'Early stopped, loading model from epoch-{stopper.best_epoch}')
                break

    if cf.early_stop > 0:
        model.load_state_dict(torch.load(cf.checkpoint_file))
    logits, adj_new, x, hid_emb, t_1_noise_adj, t_1_index = model(features, adj, mp_emb, train_x, train_y, ppr_adj, feat_score, decoder)
    cf.update(w_list[stopper.best_epoch])
    test_f1, test_mif1, val_f1, val_mif1 = eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper)
    print(f"#####DHS##result######### test_f1:{test_f1}, test_mif1:{test_mif1}########val_f1:{val_f1}, val_mif1:{val_mif1}")
    nmi_mean, nmi_std, ari_mean, ari_std = uf.evaluate_results_nc(hid_emb[test_x].detach().cpu().numpy(), test_y.cpu().numpy(), num_classes=g.n_class)
    print(f"nmi_mean:{nmi_mean}, nmi_std:{nmi_std}, ari_mean:{ari_mean}, ari_std:{ari_std}")

    # t2 = time()
    # users_to_test = list(data_generator.test_set.keys())
    # uf.link_prediction_test(hid_emb, cf.id_target_start, cf.id_target_end, args, cf, users_to_test, False, drop_flag=False, batch_test_flag=False)
    # t3 = time()

    if not log_on: uf.enable_logs()
    return cf, test_f1, test_mif1, val_f1, val_mif1


def objective(trial):

    ### 1
    params = {
        # 'dataset': trial.suggest_categorical('dataset', ['dblp','acm','yelp']),
        # 'diffusion_type': trial.suggest_categorical('diffusion_type', ['dhs','raw','ppr','hk']),
        # 'cfg': trial.suggest_categorical('cfg', [True, False]),
        # 'cfg_scale': trial.suggest_float('cfg_scale', 0, 1),
        # 'ema_scale': trial.suggest_float('ema_scale', 0, 1),
        # 'cfg_drop_rate' : trial.suggest_float('cfg_drop_rate', 0, 1),
        # 'edge_reconstr_scale': trial.suggest_float('edge_reconstr_scale', 0, 1),
        # 'l_gae_edge_rate': trial.suggest_float('l_gae_edge_rate', 0, 1),
        # 'lr': trial.suggest_float('lr', 0, 1),
        # 'weight_decay': trial.suggest_float('weight_decay', 0, 1),
    }

    ### 2
    # build args
    # args.dataset = params['dataset']
    # args.diffusion_type = params['diffusion_type']
    # args.cfg = params['cfg']
    # args.cfg_scale = params['cfg_scale']
    # args.ema_scale = params['ema_scale']
    # args.edge_reconstr_scale = params['edge_reconstr_scale']
    # args.l_gae_edge_rate = params['l_gae_edge_rate']
    # args.lr = params['lr']
    # args.weight_decay = params['weight_decay']

    ### 3
    # print(f"#############cfg_scale: {args.ema_scale}###########################################################################")
    # print(f"#############cfg_scale: {args.cfg_scale}###########################################################################")
    # print(f"#############lr: {args.lr}###########################################################################")
    # print(f"#############lr: {args.weight_decay}###########################################################################")


    _, test_f1, test_mif1, val_f1, val_mif1 = train_dhs(args, gpu_id=args.gpu_id)
    # if BEST_RESULT<test_f1:
    #     BEST_RESULT = test_f1
    #     BEST_VALUE = args.cfg_scale

    # print(f"#############result: {test_f1}###########################################################################")

    return test_f1


def build_args():
    pass

# #optuna #################################################################################################################################################
# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # dataset = 'yelp'
#     # parser.add_argument('--dataset', type=str, default=dataset)
#     # parser.add_argument('--gpu_id', type=int, default=0)
#     # parser.add_argument('--seed', type=int, default=3407)
#     # parser.add_argument('--diffusion_type', type=str, default='dhs') # raw, dhs, ppr, hk
#     # parser.add_argument('--cfg', type=bool, default=True)
#     # parser.add_argument('--cfg_scale', type=float, default=0.1)


#     # args = parser.parse_args()

#     ### 4
#     search_spase = {
#         # 'dataset':  ['dblp','acm','yelp'],
#         # 'diffusion_type': ['dhs','raw','ppr','hk'],
#         # 'cfg': [True, False],
#         # 'ema_scale': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         # 'cfg_scale': [0],
#         # 'cfg_scale': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],        
#         # 'cfg_scale': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
#         # 'cfg_scale': [0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7],
#         # 'cfg_drop_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         # 'edge_reconstr_scale': [0.00003, 0.00007, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001, 0.003, 0.005]
#         # 'l_gae_edge_rate': [0.000001, 0.000003, 0.000005, 0.000007, 0.000009], 
#         # 'lr': [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.013, 0.015, 0.017, 0.02, 0.03, 0.04, 0.05, 0.06]
#         # 'weight_decay': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0013, 0.0015, 0.0017, 0.0019]
#     }

#     study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_spase))
#     # study.optimize(objective, n_trials=100)
#     study.optimize(objective)

#     best_trial = study.best_trial
#     for key, value in best_trial.params.items():
#         print("{}: {}".format(key, value))


#     # print(f"#############value: {BEST_VALUE}####  result: {BEST_RESULT}#######################################################################")
# #optuna ##################################################################################################################################################



if __name__ == "__main__":

    train_dhs(args, gpu_id=args.gpu_id)