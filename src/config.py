from shared_configs import ModelConfig, DataConfig

e = 2.71828


class DHSConfig(ModelConfig):
    def __init__(self, dataset, seed=0):
        super(DHSConfig, self).__init__('DHS')
        default_settings = \
            {'acm': {'alpha': 1, 'dropout': 0, 'fgd_th': 0.8, 'fgh_th': 0.2, 'sem_th': 0.6,
                     'mp_list': ['psp', 'pap', 'pspap'], 'id_target_start':0, 'id_target_end':3025},
             'dblp': {'alpha': 4.5, 'dropout': 0.2, 'fgd_th': 0.99, 'fgh_th': 0.99, 'sem_th': 0.4, 'mp_list': ['apcpa'], 'id_target_start':0, 'id_target_end':2957},
             'yelp': {'alpha': 0.5, 'dropout': 0.2, 'fgd_th': 0.8, 'fgh_th': 0.1, 'sem_th': 0.2,
                      'mp_list': ['bub', 'bsb', 'bublb', 'bubsb']},
            'imdb': {'lr': 0.1, 'early_stop':80, 'weight_decay':5e-4, 'alpha': 4.5, 'dropout': 0.2, 'fgd_th': 0.99, 'fgh_th': 0.99, 'sem_th': 0.4, 'mp_list': ['m'], 'relation_type_list': ['m_d', 'm_a', 'm_d_m', 'm_a_m'], 'node_list': ['m', 'd', 'a'], 'relation_norm_list': [1, 1, 1, 1], 'relation_scale_list': [0.5, 0.5, 0.25, 0.25], 'relation_source_list': ['d', 'a', 'm', 'm'], 'relation_dropout_list': [0.15, 0.15, 0.15, 0.15, 0.15, 0.15], 'id_target_start':0, 'id_target_end':4278},
             }
        self.dataset = dataset
        self.__dict__.update(default_settings[dataset])
        # ! Model settings
        self.lr = 0.01
        self.seed = seed
        self.save_model_conf_list()  # * Save the model config list keys
        self.conv_method = 'gcn'
        self.num_head = 2
        self.early_stop = 20
        self.adj_norm_order = 1
        self.feat_norm = -1
        self.emb_dim = 64
        self.com_feat_dim = 16
        self.weight_decay = 5e-4
        self.model = 'DHS'
        self.epochs = 200
        self.exp_name = 'debug'
        self.save_weights = False
        d_conf = DataConfig(dataset)
        self.__dict__.update(d_conf.__dict__)
