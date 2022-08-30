import argparse
import pickle
import numpy as np
from baselines.DeepJIT.evaluation import evaluation_model
from baselines.DeepJIT.padding import padding_data
from baselines.DeepJIT.train import train_model
import pandas as pd
import os
import random
import torch


def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')

    parser.add_argument('-train_data', type=str, help='the directory of our training data')
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')

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
    parser.add_argument('-num_epochs', type=int, default=25, help='the number of epochs')
    parser.add_argument('-save_dir', type=str, default='', help='where to save the snapshot')
    parser.add_argument('-project_name', type=str, default='deepjit', help='save the model for project')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
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

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

        data = (pad_msg, pad_code, labels, dict_msg, dict_code)
        train_model(data=data, params=params)
    elif params.predict is True:
        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data
        labels = np.array(labels)
        ids = np.array(ids)

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
        data = (ids, pad_msg, pad_code, labels, dict_msg, dict_code)
        evaluation_model(data=data, params=params)

    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()
