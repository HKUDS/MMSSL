import pickle
import numpy as np
import torch
import torch.nn.functional as F
import scipy
# import dgl


class HIN(object):

    def __init__(self, dataset):
        data_path = f'data/{dataset}/'
        with open(f'{data_path}node_features.pkl', 'rb') as f:
            self.features = pickle.load(f)
        with open(f'{data_path}edges.pkl', 'rb') as f:
            self.edges = pickle.load(f)
        with open(f'{data_path}labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)
        with open(f'{data_path}meta_data.pkl', 'rb') as f:
            self.__dict__.update(pickle.load(f))
        if scipy.sparse.issparse(self.features):
            self.features = self.features.todense()
        # self.adj_pos = pickle.load(open(f'{data_path}all_adj','rb'))
        # self.adj_neg = pickle.load(open(f'{data_path}all_adj_neg','rb'))

        self.ppr_adj = pickle.load(open(f'{data_path}ppr_adj','rb')).cuda()
        # self.ppr_adj = None
        # if dataset=='dblp':
        #     self.n_class = 4
        # else: 
        #     self.n_class = 3

    def to_torch(self, cf):
        '''
        Returns the torch tensor of the graph.
        Args:
            cf: The ModelConfig file.
        Returns:
            features, adj: feature and adj. matrix
            mp_emb: only available for models that uses mp_list.
            train_x, train_y, val_x, val_y, test_x, test_y: train/val/test index and labels
        '''
        features = torch.from_numpy(self.features).type(torch.FloatTensor).to(cf.dev)
        train_x, train_y, val_x, val_y, test_x, test_y = self.get_label(cf.dev)

        adj = np.sum(list(self.edges.values())).todense()
        adj = torch.from_numpy(adj).type(torch.FloatTensor).to(cf.dev)
        adj = F.normalize(adj, dim=1, p=2)

        # self.ppr_adj = torch.from_numpy(self.ppr_adj).type(torch.FloatTensor).to(cf.dev)
        # adj = F.normalize(adj, dim=1, p=2)

        mp_emb = {}
        if hasattr(cf, 'mp_list'):
            for mp in cf.mp_list:
                mp_emb[mp] = torch.from_numpy(self.mp_emb_dict[mp]).type(torch.FloatTensor).to(cf.dev)
        if hasattr(cf, 'feat_norm'):
            if cf.feat_norm > 0:
                features = F.normalize(features, dim=1, p=cf.feat_norm)
                for mp in cf.mp_list:
                    mp_emb[mp] = F.normalize(mp_emb[mp], dim=1, p=cf.feat_norm)

        # self.adj_pos = dgl.from_scipy(self.adj_pos).to('cuda')
        # # G_pos = G_pos.to('cuda')
        # self.adj_neg = dgl.from_scipy(self.adj_neg).to('cuda')
        # self.adj_pos.edata['weight'] = torch.ones(len(self.adj_pos.edges()[0])).cuda()
        # self.adj_neg.edata['weight'] = torch.ones(len(self.adj_neg.edges()[0])).cuda()


        return features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y, self.ppr_adj  #, self.adj_pos, self.adj_neg

    def load_mp_embedding(self, cf):
        '''Load pretrained mp_embedding'''
        self.mp_emb_dict = {}
        for mp in cf.mp_list:
            f_name = f'{cf.data_path}{mp}_emb.pkl'
            with open(f_name, 'rb') as f:
                z = pickle.load(f)
                zero_lines = np.nonzero(np.sum(z, 1) == 0)
                if len(zero_lines) > 0:
                    # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), mode, zero_lines))
                    # z[zero_lines, :] += 1e-8
                    z[zero_lines] += 1e-8
                self.mp_emb_dict[mp] = z
        return self

    def get_label(self, dev):
        '''
        Args:
            dev: device (cpu or gpu)

        Returns:
            train_x, train_y, val_x, val_y, test_x, test_y: train/val/test index and labels
        '''
        train_x = torch.from_numpy(np.array(self.labels[0])[:, 0]).type(torch.LongTensor).to(dev)
        train_y = torch.from_numpy(np.array(self.labels[0])[:, 1]).type(torch.LongTensor).to(dev)
        val_x = torch.from_numpy(np.array(self.labels[1])[:, 0]).type(torch.LongTensor).to(dev)
        val_y = torch.from_numpy(np.array(self.labels[1])[:, 1]).type(torch.LongTensor).to(dev)
        test_x = torch.from_numpy(np.array(self.labels[2])[:, 0]).type(torch.LongTensor).to(dev)
        test_y = torch.from_numpy(np.array(self.labels[2])[:, 1]).type(torch.LongTensor).to(dev)
        return train_x, train_y, val_x, val_y, test_x, test_y
