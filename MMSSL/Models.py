import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

class MMSSL(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)             
        self.encoder = nn.ModuleDict() 
        self.encoder['image_encoder'] = self.image_trans
        self.encoder['text_encoder'] = self.text_trans

        self.common_trans = nn.Linear(args.embed_size, args.embed_size)
        nn.init.xavier_uniform_(self.common_trans.weight)
        self.align = nn.ModuleDict() 
        self.align['common_trans'] = self.common_trans

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_k': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_v': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.embed_size, args.embed_size]))),
        })
        self.embedding_dict = {'user':{}, 'item':{}}

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  
            between_sim = f(self.sim(z1[mask], z2))  

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors


    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  
       
        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.embed_size/args.head_num

        Q = torch.matmul(q, trans_w['w_q'])  
        K = torch.matmul(k, trans_w['w_k'])
        V = v

        Q = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)  
        K = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2) 
        K = torch.unsqueeze(K, 1)  
        V = torch.unsqueeze(V, 1)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=2)  

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=2)  

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        args.model_cat_rate*F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):

        image_feats = image_item_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = text_item_feats = self.dropout(self.text_trans(self.text_feats))

        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)
            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id
        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id
        user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'], self.embedding_dict['user'])
        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'], self.embedding_dict['item'])
        user_emb = user_z.mean(0)
        item_emb = item_z.mean(0)
        u_g_embeddings = self.user_id_embedding.weight + args.id_cat_rate*F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + args.id_cat_rate*F.normalize(item_emb, p=2, dim=1)

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):    
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax( torch.mm(ui_graph, i_g_embeddings) ) 
                i_g_embeddings = self.softmax( torch.mm(iu_graph, u_g_embeddings) )

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings) 
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings) 

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)


        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id



class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, int(dim/4)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim/4)),
    		nn.Dropout(args.G_drop1),

            nn.Linear(int(dim/4), int(dim/8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim/8)),
    		nn.Dropout(args.G_drop2),

            nn.Linear(int(dim/8), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = 100*self.net(x.float())  
        return output.view(-1)

