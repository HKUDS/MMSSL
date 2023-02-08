import math
import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import util_funcs 
from util_funcs import cos_sim, orthogo_tensor
import dgl


class DHS(nn.Module):
    """
    Decode neighbors of input graph.
    """
    def __init__(self, cf, g, args):
        super(DHS, self).__init__()
        self.__dict__.update(cf.get_model_conf())
        # ! Init variables
        self.dev = cf.dev
        self.ti, self.ri, self.types, self.ud_rels = g.t_info, g.r_info, g.types, g.undirected_relations
        feat_dim, mp_emb_dim = g.features.shape[1], list(g.mp_emb_dict.values())[0].shape[1]
        self.non_linear = nn.ReLU(inplace=True)
        self.cf = cf
        self.args = args
        # ! Graph Structure Learning
        MD = nn.ModuleDict
        self.fgg_direct, self.fgg_left, self.fgg_right, self.fg_agg, self.sgg_gen, self.sg_agg, self.overall_g_agg, self.overall_g_agg2 = \
            MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({}) 
        # Feature encoder
        self.encoder = MD(dict(zip(g.types, [nn.Linear(g.features.shape[1], cf.com_feat_dim) for _ in g.types])))

        if self.args.cfg==True:
            self.label_emb = nn.Embedding(g.n_class, g.n_feat)  #

        for r in g.undirected_relations:
            # ! Feature Graph Generator
            self.fgg_direct[r] = GraphGenerator(cf.com_feat_dim, self.args.num_head, cf.fgd_th, self.dev)
            self.fgg_left[r] = GraphGenerator(feat_dim, self.args.num_head, cf.fgh_th, self.dev)
            self.fgg_right[r] = GraphGenerator(feat_dim, self.args.num_head, cf.fgh_th, self.dev)
            self.fg_agg[r] = GraphChannelAttLayer(3)  # 3 = 1 (first-order/direct) + 2 (second-order)

            # ! Semantic Graph Generator
            self.sgg_gen[r] = MD(dict(
                zip(cf.mp_list, [GraphGenerator(mp_emb_dim, self.args.num_head, self.args.threshold, self.dev) for _ in cf.mp_list])))
            self.sg_agg[r] = GraphChannelAttLayer(len(cf.mp_list))

            # ! Overall Graph Generator
            self.overall_g_agg[r] = GraphChannelAttLayer(3, [1, 1, 10])  # 3 = feat-graph + sem-graph + ori_graph
            # self.overall_g_agg[r] = GraphChannelAttLayer(3)  # 3 = feat-graph + sem-graph + ori_graph
            self.overall_g_agg2[r] = GraphChannelAttLayer(2, [1, 1, 10])  # 3 = feat-graph + sem-graph + ori_graph

        # ! Graph Convolution
        if cf.conv_method == 'gcn':
            self.GCN = GCN(g.n_feat, cf.emb_dim, g.n_class, self.args.dropout)
            # self.noise_feat_GCN = GCN(g.n_feat, g.n_feat, g.n_class, cf.dropout)
        self.norm_order = self.args.adj_norm_order

    # def ppr(self, graph: nx.Graph, alpha=0.2, self_loop=True):
    def ppr(self, adj_ori, alpha=0.2, self_loop=True):
        # a = nx.convert_matrix.to_numpy_array(graph)
        a = adj_ori
        # if self_loop:
        #     a = a + torch.eye(a.shape[0]).cuda()                               # A^ = A + I_n
        # d = torch.diag(torch.sum(a, 1))                                     # D^ = Sigma A^_ii
        # dinv_left = torch.diag(torch.float_power(torch.sum(a, 1)+1e-8, -0.5).flatten()).float()                                    # D^ = Sigma A^_ii
        # dinv_right = torch.diag(torch.float_power(torch.sum(a.T, 1)+1e-8, -0.5).flatten()).float()                                    # D^ = Sigma A^_ii
        dinv = torch.diag(torch.float_power(torch.sum(a, 1)+1e-8, -0.5).flatten()).float()                                    # D^ = Sigma A^_ii

        # dinv = torch.linalg.matrix_power(d, -0.5)                       # D^(-1/2)
        # dinv = torch.matrix_power(d, -0.5)                       # D^(-1/2)
        # at = torch.matmul(torch.matmul(dinv_left, a), dinv_right)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
        at = torch.matmul(torch.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
        new_adj = alpha * torch.linalg.inv((torch.eye(a.shape[0]).cuda() - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
        # new_adj = alpha * torch.linalg.inv(( - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

        # new_adj = torch.from_numpy(new_adj).cuda()
        # new_adj1 = new_adj.type(torch.float16)
        new_adj = at.float()
        return new_adj

    # def heatkernel(self, graph: nx.Graph, t=5, self_loop=True):
    def heatkernel(self, adj_ori, t=5, self_loop=True):
        # a = nx.convert_matrix.to_numpy_array(graph)
        a = adj_ori.cpu().numpy()
        if self_loop:
            a = a + np.eye(a.shape[0])
        d = np.diag(np.sum(a, 1))
        new_adj = np.exp(t * (np.matmul(a, inv(d)) - 1))
        return torch.from_numpy(new_adj).float().cuda()


    def dhs_ppr(self, features, adj_ori, mp_emb, adj_ppr):
        def get_rel_mat(mat, r):
            return mat[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]]

        def get_type_rows(mat, type):
            return mat[self.ti[type]['ind'], :]

        def gen_g_via_feat(graph_gen_func, mat, r):
            return graph_gen_func(get_type_rows(mat, r[0]), get_type_rows(mat, r[-1]))

        # ! Heterogeneous Feature Mapping
        com_feat_mat = torch.cat([self.non_linear(
            self.encoder[t](features[self.ti[t]['ind']])) for t in self.types])

        # ! Heterogeneous Graph Generation
        new_adj = torch.zeros_like(adj_ori).to(self.dev)
        for r in self.ud_rels:
            ori_g = get_rel_mat(adj_ori, r)
            ppr_g = get_rel_mat(adj_ppr, r)

            # ! Semantic Graph Generation
            sem_g_list = [gen_g_via_feat(self.sgg_gen[r][mp], mp_emb[mp], r) for mp in mp_emb]
            sem_g = self.sg_agg[r](sem_g_list)
            # ! Overall Graph
            # Update relation sub-matixs
            # sem_g = sem_g/(sem_g.max()/ppr_g.max())
            # ppr_g = F.sigmoid(ppr_g)
            # sem_g = sem_g + sem_g.t()
            ppr_g = F.normalize(ppr_g, dim=0, p=2)
            if self.args.if_ori_g:
                new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = \
                    self.overall_g_agg[r]([sem_g, ppr_g, ori_g])  # update edge  e.g. AP
            # new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = torch.multiply(ppr_g, sem_g)  # update edge  e.g. AP
            else:
                new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = \
                    self.overall_g_agg2[r]([sem_g, ppr_g]) 

        new_adj = new_adj + new_adj.t()  # sysmetric  [3913]

        return new_adj   

    def noise_drop(self, noise_data, t_noise, noise_new_adj, noise_index_x, noise_index_y, adj_ori, new_adj):
        _, t_index = torch.topk(noise_data, t_noise, largest=False)
        # t_index = t_index.half() 
        t_noise_adj = noise_new_adj
        t_noise_adj[noise_index_x[t_index],noise_index_y[t_index]] = 0
        t_noise_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]] +=  new_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]]
        return t_noise_adj


    def get_noise_adj(self, new_adj, adj_ori):

        #########################################################################################
        # # print(f"new adj sum: {new_adj.sum()} ")
        # time step:
        # get noise edge:
        noise_new_adj = new_adj.clone().detach()
        noise_new_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]] = 0
        # get num of noise edge:
        total_data = torch.where(noise_new_adj!=0)[0].shape[0]
        func = util_funcs.get_pacing_function(total_step=self.args.total_step, total_data=total_data, pacing_f='linear') 
        #sample time step t
        t = torch.randint(low=1, high=self.args.total_step, size=(1,))
        t_noise, t_1_noise = func(t), func(t-1)#
        t_noise = total_data - t_noise
        t_1_noise = total_data - t_1_noise 
        # t_1_noise = func(t-1)
        noise_index_x, noise_index_y = torch.where(noise_new_adj!=0)[0],torch.where(noise_new_adj!=0)[1]
        # noise_index_x, noise_index_y = noise_index_x.half(), noise_index_y.half()
        noise_data = noise_new_adj[noise_index_x, noise_index_y]
        # #---
        # _, t_index = torch.topk(noise_data, t_noise, largest=False)
        # # t_index = t_index.half() 
        # t_noise_adj = noise_new_adj
        # t_noise_adj[noise_index_x[t_index],noise_index_y[t_index]] = 0
        # t_noise_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]] +=  new_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]]
        # #---
        t_noise_adj =  self.noise_drop(noise_data, t_noise, noise_new_adj, noise_index_x, noise_index_y, adj_ori, new_adj)
        # #---
        # _, t_1_index = torch.topk(noise_data, t_1_noise, largest=False) 
        # # t_1_index = t_1_index.half()
        # t_1_noise_adj = noise_new_adj
        # t_1_noise_adj[noise_index_x[t_1_index],noise_index_y[t_1_index]] = 0
        # t_1_noise_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]] +=  new_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]]
        # # ---
        t_1_noise_adj =  self.noise_drop(noise_data, t_1_noise, noise_new_adj, noise_index_x, noise_index_y, adj_ori, new_adj)

        # ! Aggregate
        # adj = t_noise_adj 
        # adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]] +=  new_adj[torch.where(adj_ori!=0)[0],torch.where(adj_ori!=0)[1]]
        # adj = F.normalize(adj, dim=0, p=self.norm_order)
        # logits, x, hid_emb = self.GCN(features, adj)
        # logits, x, hid_emb = self.GCN(input_features, new_adj)
        new_adj[torch.where(t_noise_adj==0)[0],torch.where(t_noise_adj==0)[1]] = 0
        #########################################################################################
        return new_adj, t_1_noise_adj, t

    def get_noise_node_index(self, tmp_num_nodes, mask_rate=0.3):
        # perm = torch.randperm(tmp_num_nodes, device=x.device)
        perm = torch.randperm(tmp_num_nodes)
        num_mask_nodes = int(mask_rate * tmp_num_nodes)

        # random masking
        # num_mask_nodes = int(mask_rate * tmp_num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        return mask_nodes, keep_nodes

    def get_noise_feat(self, t_index, features, new_adj, t,  beta_start=1e-4, beta_end=0.02):
        # # _, _, new_feat = self.noise_feat_GCN(features, new_adj)
        # noise = torch.randn_like(features)
        # temAlphas = self.alphasProdSqrt[t]
        # temStd = self.oneMinusAlphasBarSqrt[t]
        # noisy_feat = (temAlphas * features + temStd * noise)
        # features[t_index] = noisy_feat[t_index]

		# self.alphasProdSqrt = t.sqrt(alphasProd).cuda()
		# self.oneMinusAlphasBarSqrt = t.sqrt(1 - alphasProd).cuda()


        beta = torch.linspace(beta_start, beta_end, self.args.total_step).cuda()
        alpha = 1. - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        # sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]

        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(features)
        noisy_feat = sqrt_alpha_hat * features + sqrt_one_minus_alpha_hat * Ɛ

        mask_nodes, keep_nodes = self.get_noise_node_index(features.shape[0], self.args.feature_mask_rate)
        features[mask_nodes] = noisy_feat[mask_nodes]
        return features

    def forward(self, input_features, adj_ori, mp_emb, train_x, train_y, ppr_adj, feat_score, decoder):  #[3913, 82] [3913, 3913] {[3913, 64], [3913, 64], [3913, 64], [3913, 64]}=====
        """
            Step1: diffusion
            Step2: generation
        """
        features = input_features
        if self.args.cfg==True:
            # orthogo_tensor(self.label_emb.weight)
            features[train_x] = (1-self.args.cfg_scale)*features[train_x] + self.args.cfg_scale*self.label_emb(train_y).data
            # input_features[train_x] = (1-self.args.cfg_scale)*features[train_x] + self.args.cfg_scale*self.label_emb(train_y).data


        if self.args.diffusion_type=='raw':
            noise_adj = adj_ori
        elif self.args.diffusion_type=='dhs':
            noise_adj = self.dhs(features, adj_ori, mp_emb)
        elif self.args.diffusion_type=='ppr':
            noise_adj = ppr_adj
            # new_adj = self.ppr(adj_ori)
            # pickle.dump(new_adj, open('/home/ww/FILE_from_ubuntu18/Code/work7/KDD23HetoDiffusion/DHS/data/acm/ppr_adj','wb'))
        elif self.args.diffusion_type=='hk':
            noise_adj = self.heatkernel(adj_ori, t=5, self_loop=True)
        elif self.args.diffusion_type=='dhs_ppr':
            noise_adj = self.dhs_ppr(features, adj_ori, mp_emb, ppr_adj)
        elif self.args.diffusion_type=='ppr_dhs':
            # new_adj = self.dhs_ppr(features, adj_ori, mp_emb, ppr_adj)
            noise_adj = self.dhs(features, adj_ori, mp_emb)
            noise_adj = self.ppr(noise_adj)

        # elif self.args.diffusion_type=='hk':
        #     pass
        # elif self.args.diffusion_type=='hk':
        #     pass

        # if self.args.cfg==True:
        #     features[train_x] = features[train_x] + self.label_emb(train_y).data

        # x_index, y_index = torch.where(new_adj!=0)[0], torch.where(new_adj!=0)[1] 
        # g = dgl.graph((x_index, y_index)).to('cuda:0')
        # g.edata['weight'] = new_adj[x_index, y_index]
        # node_id = torch.arange(new_adj.shape[0]).cuda()
        # node_id = torch.arange( self.cf.id_target_end ).cuda()
        # # random_walk:
        # sub_graph = dgl.sampling.random_walk(g, node_id, length=5)
        # #node2vec_radom_walk:
        # g = g.to('cpu')
        # node_id = node_id.cpu()
        # sub_graph = dgl.sampling.node2vec_random_walk(g, node_id, p=0.4, q=1, walk_length=5)
        # # neighbor:
        # sub_graph = dgl.sampling.sample_neighbors(g, node_id, fanout=5)
        # # #neighbor_biased:
        # # sub_graph = dgl.sampling.sample_neighbors_biased(g, node_id, fanout=5, )
        # #select_topk:
        # g = g.to('cpu')
        # node_id = node_id.cpu()
        # sub_graph = dgl.sampling.select_topk(g, k=5, weight='weight', nodes=node_id)

        t_noise_adj, t_1_noise_adj, t = self.get_noise_adj(noise_adj, adj_ori)
        torch.cuda.empty_cache()

        # t_index, t_1_index = util_funcs.get_noise_feat_index(feat_score, features.shape[0], t, self.args.total_step)
        # noise_features = self.get_noise_feat(t_index, features, noise_adj, t)

        # noise_adj = F.normalize(noise_adj, dim=0, p=self.norm_order)
        noise_adj = F.normalize(t_noise_adj, dim=0, p=self.norm_order)

        logits, x, hid_emb = self.GCN(features, noise_adj)
        # logits, x, hid_emb = self.GCN(noise_features, noise_adj)
        torch.cuda.empty_cache()


        # return logits, noise_adj, x, hid_emb, t_1_noise_adj, t_1_index
        return logits, noise_adj, x, hid_emb, t_1_noise_adj, None


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj), inplace=True)
        hid_emb = F.dropout(x, self.dropout, training=self.training)  #[3913, 64]
        x = self.gc2(hid_emb, adj)  #[3913, 3]
        return F.log_softmax(x, dim=1), x, hid_emb
        # return x


class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        # support = support.type(torch.float16)
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# class EMA:
#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta
#         self.step = 0

#     def update_model_average(self, ma_model, current_model):
#         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#             old_weight, up_weight = ma_params.data, current_params.data
#             ma_params.data = self.update_average(old_weight, up_weight)

#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new

#     def step_ema(self, ema_model, model, step_start_ema=2000):
#         if self.step < step_start_ema:
#             self.reset_parameters(ema_model, model)
#             self.step += 1
#             return
#         self.update_model_average(ema_model, model)
#         self.step += 1

#     def reset_parameters(self, ema_model, model):
#         ema_model.load_state_dict(model.state_dict())



class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()




class GCN_trans_layer(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_trans_layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(nfeat, nhid))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = dropout

    def forward(self, x, adj):
        support = F.dropout(torch.spmm(x, self.weight), self.dropout)  # HW in GCN
        # support = F.dropout(torch.spmm(x, self.weight), self.dropout)
        output = torch.spmm(adj, support)  # AHW

        # support = torch.spmm(adj, x)  # AHW
        # output = torch.spmm(support, self.weight)  # HW in GCN
        # # output = torch.spmm(adj, support)  # AHW

        # if self.bias is not None:
        #     return output + self.bias
        # else:
        return output


class GCN_layer(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(nhid, nhid))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = dropout

    def forward(self, x, adj):
        # support = torch.spmm(x, self.weight)  # HW in GCN
        # output = torch.spmm(adj, support)  # AHW

        support = torch.spmm(adj, x)  # AHW
        output = torch.spmm(support, self.weight)  # HW in GCN
        # if self.bias is not None:
        #     return output + self.bias
        # else:
        return output






class Decoder(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, feat, cf, g):
        super(Decoder, self).__init__()
        self.__dict__.update(cf.get_model_conf())
        # ! Init variables
        self.dataset = cf.dataset
        self.dev = cf.dev
        # self.node_type = cf.node_list
        # self.relation_type = cf.relation_type_list
        # self.relation_source = cf.relation_source_list
        # self.relation_norm = cf.relation_norm_list
        # self.relation_scale = cf.relation_scale_list
        # self.relation_dropout = cf.relation_dropout_list

        # self.features = g.features 
        self.emb_dim = cf.emb_dim
        self.non_linear = nn.ReLU(inplace=True)

        # feature & embedding
        self.features = g.features
        # self.embedddings_weights = self.init_embedding(cf, g)
        self.reduce_weight = nn.Parameter(torch.Tensor(cf.emb_dim, g.n_class))
        nn.init.xavier_uniform_(self.reduce_weight)
        self.trans_weight = nn.Parameter(torch.Tensor(g.features.shape[1] , cf.emb_dim))
        nn.init.xavier_uniform_(self.trans_weight)

        # ! Graph Convolution
        if cf.dataset == 'yelp':
            self.GCN_trans_layers = torch.nn.ModuleList()
            for id, t_type in enumerate(self.relation_type): 
                self.GCN_trans_layers.append(GCN_trans_layer(cf.emb_dim, g.n_feat, self.relation_dropout[id]))
                # self.GCN_layer = GCN_layer(cf.emb_dim, cf.emb_dim, cf.dropout)
        else:
            self.GCN_trans_layers = GCN_trans_layer(cf.emb_dim, g.n_feat, cf.dropout)


    def forward(self, features, adj_ori):  #, mp_emb):

        # # ! Heterogeneous Graph Generation
        # anchor_embs, nei_embs = [], []
        # feat_learned = dict()

        # for id, t_type in enumerate(self.relation_type): 
        #     if self.dataset == 'yelp':
        #         emb_anchor, emb_nei = self.GCN_trans_layers[id](features[self.relation_source[id]], adj_ori[id])
        #         feat_learned[self.relation_source[id]] = emb_nei
        #         # if self.relation_norm[id]:
        #         #     anchor_embs.append(  self.relation_scale[id]*F.normalize(emb_anchor , p=2, dim=1))
        #         #     nei_embs.append( self.relation_scale[id]*F.normalize(emb_nei , p=2, dim=1))
        #         # else:
        #         #     anchor_embs.append(self.relation_scale[id]*emb_anchor)
        #         #     nei_embs.append(self.relation_scale[id]*emb_nei)
        #     else:
        #         emb_anchor, emb_nei = self.GCN_trans_layers(features[self.relation_source[id]], adj_ori[id])
        #         feat_learned[self.relation_source[id]] = emb_nei
        #         # if self.relation_norm[id]:
        #         #     anchor_embs.append(  self.relation_scale[id]*F.normalize(emb_anchor , p=2, dim=1))
        #         #     nei_embs.append( self.relation_scale[id]*F.normalize(emb_nei , p=2, dim=1))
        #         # else:
        #         #     anchor_embs.append(  self.relation_scale[id]*emb_anchor)
        #         #     nei_embs.append(  self.relation_scale[id]*emb_nei)

        # # ! Aggregate
        # # anchor_embs_agg = torch.mean(torch.stack(anchor_embs), dim=0)
        # # output = torch.mm(anchor_embs_agg, self.reduce_weight)  # HW in GCN
        # # return F.log_softmax(output, dim=1), anchor_embs_agg, nei_embs, feat_learned

        re_feature = self.GCN_trans_layers(features, adj_ori)

        return re_feature




