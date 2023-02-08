import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import dgl

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC


from sympy.matrices import Matrix, GramSchmidt
import multiprocessing

import metrics
from load_data import Data
# from utility.parser import parse_args
import parser


import heapq

cores = multiprocessing.cpu_count() // 5

# args = parser.parse_args()
# data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)


def loss_function(pred, drop_rate):
    # loss = F.cross_entropy(y, t, reduce = False)
    # loss_mul = loss * t
    ind_sorted = np.argsort(pred.cpu().data).cuda()
    loss_sorted = pred[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = pred[ind_update]

    return loss_update.mean()



# def loss_function(y, t, drop_rate):
#     # loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
#     loss = F.cross_entropy(y, t, reduce = False)

#     loss_mul = loss * t
#     ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
#     loss_sorted = loss[ind_sorted]

#     remember_rate = 1 - drop_rate
#     num_remember = int(remember_rate * len(loss_sorted))

#     ind_update = ind_sorted[:num_remember]

#     # loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])
#     loss_update = F.cross_entropy(y[ind_update], t[ind_update])

#     return loss_update


def get_pacing_function(total_step=1000, total_data=10000, pacing_a=1, pacing_b=0, pacing_f='linear'):
    """Return a  pacing function  w.r.t. step.
    input:
    a:[0,large-value] percentage of total step when reaching to the full data. This is an ending point (a*total_step, total_data)) 
    b:[0,1]  percentatge of total data at the begining of the training. Thia is a starting point (0,b*total_data))
    """
    a = pacing_a  #0.8
    b = pacing_b  #0.2
    index_start = b*total_data
    if pacing_f == 'linear':
      rate = (total_data - index_start)/(a*total_step)
      def _linear_function(step):
        return int(rate *step + index_start)
      return _linear_function
    
    elif pacing_f == 'quad':
      rate = (total_data-index_start)/(a*total_step)**2  
      def _quad_function(step):
        return int(rate*step**2 + index_start)
      return _quad_function
    
    elif pacing_f == 'root':
      rate = (total_data-index_start)/(a*total_step)**0.5
      def _root_function(step):
        return int(rate *step**0.5 + index_start)
      return _root_function
    
    # elif pacing_f == 'step':
    #   threshold = a*total_step
    #   def _step_function(step):
    #     # return int( total_data*(step//threshold) +index_start)
    #     return int( ((total_data*step)//threshold) +index_start)
    #   return _step_function      

    elif pacing_f == 'exp':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      rate =  (total_data-tilde_b)/(np.exp(c)-1)
      constant = c/tilde_a
      def _exp_function(step):
        if not np.isinf(np.exp(step *constant)):
            return int(rate*(np.exp(step*constant)-1) + tilde_b )
        else:
            return total_data
      return _exp_function

    elif pacing_f == 'log':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      ec = np.exp(-c)
      N_b = (total_data-tilde_b)
      def _log_function(step):
        return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
      return _log_function




def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_node(x, item_num, Ks, args):
    # user u's ratings for user u
    is_val = x[-1]
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    if is_val:
        user_pos_test = data_generator.val_set[u]
    else:
        user_pos_test = data_generator.test_set[u]

    all_items = set(range(item_num))

    test_items = list(all_items - set(training_items))

    # if args.test_flag == 'part':
    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    # else:
    #     r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def link_prediction_test(embed, id_target_start, id_target_end, args, cf, users_to_test, is_val, drop_flag=False, batch_test_flag=False):
    Ks = eval(args.Ks)
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    pool = multiprocessing.Pool(cores)

    u_batch_size = args.lp_batch_size * 2
    i_batch_size = args.lp_batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start: end]
        if batch_test_flag:
            n_item_batchs = id_target_end // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), id_target_end))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, id_target_end)

                item_batch = range(i_start, i_end)
                u_g_embeddings = embed[id_target_start: id_target_end][user_batch]
                i_g_embeddings = embed[item_batch]
                i_rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == id_target_end
        else:
            item_batch = range(id_target_end)
            u_g_embeddings = embed[id_target_start: id_target_end][user_batch]
            i_g_embeddings = embed[item_batch]
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

        rate_batch = rate_batch.detach().cpu().numpy()
        user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))

        batch_result = pool.map(test_one_node, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result



def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def evaluate_results_nc(embeddings, labels, num_classes):
    # print('SVM test')
    # svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    # print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
    #                                 (macro_f1_mean, macro_f1_std), train_size in
    #                                 zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    # print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
    #                                 (micro_f1_mean, micro_f1_std), train_size in
    #                                 zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)  #(3478, 512) (3478)  3
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    # return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std
    return nmi_mean, nmi_std, ari_mean, ari_std




def orthogo_tensor(x):
    m, n = x.size()
    x_np = x.t().cpu().detach().numpy()
    matrix = [Matrix(col) for col in x_np.T]
    gram = GramSchmidt(matrix)
    ort_list = []
    for i in range(m):
        vector = []
        for j in range(n):
            vector.append(float(gram[i][j]))
        ort_list.append(vector)
    ort_list = np.mat(ort_list)
    ort_list = torch.from_numpy(ort_list)
    ort_list = F.normalize(ort_list,dim=1)
    x.weight = ort_list
    return x




def mse_criterion(x, y, mask_nodes_dict=None, alpha=3):

    # res_list = []
    # for id, value in enumerate(x_dict):
    #     # x, y  = x_dict[value][mask_nodes_dict[value]], y_dict[value][mask_nodes_dict[value]]
    # x, y  = x_dict[value], y_dict[value]

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)
    tmp_loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    tmp_loss = tmp_loss.mean()

    loss = F.mse_loss(x, y)
    # res_list.append(tmp_loss)
    # loss = sum(res_list)/len(res_list)
    return loss


def sce_criterion(x, y, alpha=1, tip_rate=0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1-(x*y).sum(dim=-1)).pow_(alpha)


    if tip_rate!=0:
        loss = loss_function(loss, tip_rate)   
        return loss

    loss = loss.mean() 

    # loss = loss.mean()

    return loss


def get_feat_score(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    score = torch.sum(torch.mul(x,y), dim=1)
    return score

def get_noise_feat_index(feat_score, node_num, t, total_step):
    # feat_score = get_feat_score(x, y)

    func = get_pacing_function(total_step=total_step, total_data=node_num, pacing_f='linear') 
    t_noise, t_1_noise = func(t), func(t-1)#

    t_value, t_index = torch.topk(feat_score, t_noise, largest=False)
    t_1_value, t_1_index = torch.topk(feat_score, t_1_noise, largest=False)

    return t_index, t_1_index

def sample_timesteps(noise_steps, n):
    return torch.randint(low=1, high=noise_steps, size=(n,))


# def dgl_sample(g_pos, g_neg, samp_num, samp_num_neg, anchor_id):  #, node_type, edge_type): #

#     pos_row_list = []
#     pos_col_list = []
#     neg_row_list = []
#     neg_col_list = []    

#     #pos
#     sub_g_pos = dgl.sampling.sample_neighbors(g_pos, anchor_id, samp_num, edge_dir='out', replace=True) #
#     # sub_neg_g = dgl.sampling.global_uniform_negative_sampling(g, samp_num_neg)
#     sub_g_neg = dgl.sampling.sample_neighbors(g_neg, anchor_id, samp_num_neg, edge_dir='out', replace=True) #

#     # sub_pos_g = dgl.sampling.select_topk(g_pos.cpu(), nodes={node_type: anchor_id.cpu()}, k=samp_num, weight='weight') #

#     row_pos, col_pos = sub_g_pos.edges()
#     row_neg, col_neg = sub_g_neg.edges()

#     pos_row_list = row_pos.reshape(len(row_pos), samp_num)
#     pos_col_list = col_pos .reshape(len(row_pos), samp_num)
#     neg_row_list = row_neg.reshape(len(row_pos), samp_num_neg)
#     neg_col_list = col_neg.reshape(len(row_pos), samp_num_neg)
   
#     return pos_row_list, pos_col_list, neg_row_list, neg_col_list   


def innerProduct(u, i, u_id=None, j_id=None):  #
    pred_i = torch.sum(torch.mul(u,i), dim=1)  #*args.mult   #[4096, 7]==>[4096]
    if u_id==None:
        return pred_i
    else:
        pred_j = torch.sum(torch.mul(u_id,j_id), dim=1)  #*args.mult
        return pred_i, pred_j  #[4096, ***]  [4096, 15]


# * ============================= Init =============================
def shell_init(server='S5', gpu_id=0):
    '''

    Features:
    1. Specify server specific source and python command
    2. Fix Pycharm LD_LIBRARY_ISSUE
    3. Block warnings
    4. Block TF useless messages
    5. Set paths
    '''
    import warnings
    np.seterr(invalid='ignore')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if server == 'Xy':
        python_command = '/home/chopin/zja/anaconda/bin/python'
    elif server == 'Colab':
        python_command = 'python'
    else:
        python_command = '~/anaconda3/bin/python'
        if gpu_id > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'  # Extremely useful for Pycharm users
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Block TF messages

    return python_command


def seed_init(seed):
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    dgl.random.seed(seed)


# * ============================= Torch =============================

def exists_zero_lines(h):
    zero_lines = torch.where(torch.sum(h, 1) == 0)[0]
    if len(zero_lines) > 0:
        # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), 'emb', zero_lines))
        print(f'{len(zero_lines)} zero lines !\nZero lines:{zero_lines}')
        return True
    return False


def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


# * ============================= Print Related =============================

def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def block_logs():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def progress_bar(prefix, start_time, i, max_i, postfix):
    """
    Generates progress bar AFTER the ith epoch.
    Args:
        prefix: the prefix of printed string
        start_time: start time of the loop
        i: finished epoch index
        max_i: total iteration times
        postfix: the postfix of printed string

    Returns: prints the generated progress bar

    """
    cur_run_time = time.time() - start_time
    i += 1
    if i != 0:
        total_estimated_time = cur_run_time * max_i / i
    else:
        total_estimated_time = 0
    print(
        f'{prefix} :  {i}/{max_i} [{time2str(cur_run_time)}/{time2str(total_estimated_time)}, {time2str(total_estimated_time - cur_run_time)} left] - {postfix}-{get_cur_time()}')


def print_train_log(epoch, dur, loss, train_f1, val_f1):
    print(
        f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss.item():.4f} | TrainF1 {train_f1:.4f} | ValF1 {val_f1:.4f}")


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= File Operations =============================

def write_nested_dict(d, f_path):
    def _write_dict(d, f):
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write(str(d[key]) + '\n')

    with open(f_path, 'a+') as f:
        f.write('\n')
        _write_dict(d, f)


def save_pickle(var, f_name):
    pickle.dump(var, open(f_name, 'wb'))


def load_pickle(f_name):
    return pickle.load(open(f_name, 'rb'))


def clear_results(dataset, model):
    res_path = f'results/{dataset}/{model}/'
    os.system(f'rm -rf {res_path}')
    print(f'Results in {res_path} are cleared.')


# * ============================= Path Operations =============================

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    # python 中的文件目录对空格的处理为空格，命令行对空格的处理为'\ '所以命令行相关需 replace(' ','\ ')
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    elif style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:

        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = os.path.dirname(p)
        mkdir_p(p, log)


# * ============================= Time Related =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time():
    import datetime
    dt = datetime.datetime.now()
    return f'{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}'


# * ============================= Others =============================
def print_weights(model, interested_para='_agg'):
    w_dict = {}
    for name, W in model.named_parameters():
        if interested_para in name:
            data = F.softmax(W.data.squeeze()).cpu().numpy()
            # print(f'{name}:{data}')
            w_dict[name] = data
    return w_dict


def count_avg_neighbors(adj):
    return len(torch.where(adj > 0)[0]) / adj.shape[0]


