import util_funcs as uf


class ModelConfig():
    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.model_conf_list = None

    def __str__(self):
        # Print all attributes including data and other path settings added to the config object.
        return str(self.__dict__)

    def save_model_conf_list(self):
        self.model_conf_list = list(self.__dict__.copy().keys())
        self.model_conf_list.remove('model_conf_list')

    def update_file_conf(self):
        f_conf = FileConfig(self)
        self.__dict__.update(f_conf.__dict__)
        return self

    def model_conf_to_str(self):
        # Print the model settings only.
        return str({k: self.__dict__[k] for k in self.model_conf_list})

    def get_model_conf(self):
        # Return the model settings only.
        return {k: self.__dict__[k] for k in self.model_conf_list}

    def update(self, conf_dict):
        self.__dict__.update(conf_dict)
        self.update_file_conf()
        return self


class DataConfig:
    def __init__(self, dataset):
        data_conf = {
            'acm': {'data_type': 'pas', 'relation_list': 'p-a+a-p+p-s+s-p'},
            'dblp': {'data_type': 'apc', 'relation_list': 'p-a+a-p+p-c+c-p'},
            'imdb': {'data_type': 'mad', 'relation_list': 'm-a+a-m+m-d+d-m'},
            'aminer': {'data_type': 'apr', 'relation_list': 'p-a+p-r+a-p+r-p'},
            'yelp': {'data_type': 'busl', 'relation_list': 'b-u+u-b+b-s+s-b+b-l+l-b'}
        }
        self.__dict__.update(data_conf[dataset])
        self.dataset = dataset
        self.data_path = f'data/{dataset}/'

        return


class FileConfig:

    def __init__(self, cf: ModelConfig):
        '''
            1. Set f_prefix for each model. The f_prefix stores the important hyperparamters (tuned parameters) of the model.
            2. Generate the file names using f_prefix.
            3. Create required directories.
        '''
        # if cf.model[:4] == 'DHS':
        f_prefix = f'do{cf.dropout}_lr{cf.lr}_a{cf.alpha}_tr{cf.fgd_th}-{cf.fgh_th}-{cf.sem_th}_mpl{uf.mp_list_str(cf.mp_list)}'
        self.res_file = f'results/{cf.dataset}/{cf.model}/{cf.exp_name}/<{cf.model}>{f_prefix}.txt'
        self.checkpoint_file = f'temp/{cf.model}/{cf.dataset}/{f_prefix}{uf.get_cur_time()}.pt'
        uf.mkdir_list(self.__dict__.values())
