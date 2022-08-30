import torch
import os
from baselines.utils.performance_measure import PerformanceMeasure
from baselines.utils.results_writer import ResultWriter
from sklearn import preprocessing
import time
from baselines.Deeper.LR import LR
from baselines.Deeper.DBN import DBN
import numpy as np
import math
import random
import torch.nn as nn
from baselines.utils.preprocess_data import load_data, load_test_dataframe
import warnings

warnings.filterwarnings("ignore")

colomn_names = ['project', 'parent_hashes', 'commit_hash', 'author_name',
                'author_email', 'author_date', 'author_date_unix_timestamp',
                'commit_message', 'la', 'ld', 'fileschanged', 'nf', 'ns', 'nd',
                'entropy', 'ndev', 'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp',
                'classification', 'fix', 'is_buggy_commit']
feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
label_name = ["is_buggy_commit"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    # np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_update(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    # np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1.0]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0.0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def DBN_JIT(train_features, train_labels, test_features, test_labels, hidden_units=[20, 12, 12], num_epochs_LR=200):
    # training DBN model
    #################################################################################################
    starttime = time.time()
    dbn_model = DBN(visible_units=train_features.shape[1],
                    hidden_units=hidden_units,
                    use_gpu=False)
    dbn_model.train_static(train_features, train_labels, num_epochs=10)
    # Finishing the training DBN model
    # print('---------------------Finishing the training DBN model---------------------')
    # using DBN model to construct features
    DBN_train_features, _ = dbn_model.forward(train_features)
    DBN_test_features, _ = dbn_model.forward(test_features)
    DBN_train_features = DBN_train_features.numpy()
    DBN_test_features = DBN_test_features.numpy()

    train_features = np.hstack((train_features, DBN_train_features))
    test_features = np.hstack((test_features, DBN_test_features))
    if len(train_labels.shape) == 1:
        num_classes = 1
    else:
        num_classes = train_labels.shape[1]
    # lr_model = LR(input_size=hidden_units, num_classes=num_classes)
    lr_model = LR(input_size=train_features.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.00001)
    steps = 0
    batches_test = mini_batches(X=test_features, Y=test_labels)
    for epoch in range(1, num_epochs_LR + 1):
        # building batches for training model
        batches_train = mini_batches_update(X=train_features, Y=train_labels)
        for batch in batches_train:
            x_batch, y_batch = batch
            x_batch, y_batch = torch.tensor(x_batch).float(), torch.tensor(y_batch).float()

            optimizer.zero_grad()
            predict = lr_model.forward(x_batch)
            loss = nn.BCELoss()
            loss = loss(predict, y_batch)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 50 == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

    y_pred_prob, lables = lr_model.predict(data=batches_test)

    return y_pred_prob


def DBN_train_and_eval(baseline_name: str):
    set_seed(seed=42)
    X_train, y_train, X_test, y_test = load_data(base_path, baseline_name)
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)

    print(f"building model {baseline_name}")
    y_pred_prob = DBN_JIT(X_train, y_train, X_test, y_test)

    result_df = load_test_dataframe(base_path, baseline_name)
    result_df["defective_commit_prob"] = y_pred_prob
    result_df["defective_commit_pred"] = [1.0 if p >= 0.5 else 0.0 for p in y_pred_prob]

    presults = PerformanceMeasure().eval_metrics(result_df=result_df)
    print(presults)
    ResultWriter().write_result(result_path=result_path, method_name="Deeper",
                                presults=presults)


if __name__ == "__main__":
    print("Running deeper model")
    base_path = "data/"
    result_path = os.path.dirname(os.path.dirname(__file__)) + '/results/'
    DBN_train_and_eval('deeper')
