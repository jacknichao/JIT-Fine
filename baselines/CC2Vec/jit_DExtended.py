import argparse
from baselines.CC2Vec.jit_DExtended_padding import padding_data
import pickle
import numpy as np
from baselines.CC2Vec.jit_DExtended_eval import evaluation_model
from baselines.CC2Vec.jit_DExtended_train import train_model
import torch
import random

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-project', type=str, default='', help='name of the dataset')
    # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')

    parser.add_argument('-train_data', type=str, help='the directory of our training data')
    parser.add_argument('-train_data_cc2ftr', type=str, help='the directory of our training data with cc2ftr')
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')
    parser.add_argument('-pred_data_cc2ftr', type=str, help='the directory of our testing data with cc2ftr')

    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')

    # CUDA
    parser.add_argument('-device', type=int, default=0,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    params = read_args().parse_args()
    set_seed(seed=42)
    if params.train is True:
        data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = data
        labels = np.array(labels)

        data_ftr = pickle.load(open(params.train_data_cc2ftr, 'rb'))

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

        data = (data_ftr, pad_msg, pad_code, labels, dict_msg, dict_code)

        # start = time.time()
        train_model(data=data, params=params)
        # end = time.time()
        # train_time_sec = end-start

        # with open(params.project+'_DeepJIT_train_time.txt','a+') as f:
        #     f.write(str(train_time_sec)+'\n')


    elif params.predict is True:
        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data
        labels = np.array(labels)
        ids = np.array(ids)

        # # 还需要增加对应的la +ld
        # pname = os.path.basename(params.pred_data).split('_')[0]
        # csv_files = pd.read_csv( os.path.join(
        #                         os.path.dirname(params.pred_data),pname+"_k_feature.csv"
        #                         ))
        # csv_files = csv_files[csv_files['commit_id'].isin(ids)]
        # csv_files['LOC'] = csv_files['la'] + csv_files['ld']
        # LOCs = csv_files['LOC'].to_list()
        # # 增加id，以确定是不是对应的commit
        # ids = np.array(ids)
        # LOCs = np.array(LOCs)

        data_ftr = pickle.load(open(params.pred_data_cc2ftr, 'rb'))

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

        # 增加了id 和 LOCs
        data = (ids, data_ftr, pad_msg, pad_code, labels, dict_msg, dict_code)
        evaluation_model(data=data, params=params)
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()
