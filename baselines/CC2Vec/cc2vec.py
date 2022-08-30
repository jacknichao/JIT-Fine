import os

cc2ftr_train = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_cc2ftr -train -train_data data/cc2vec/features_train.pkl  -dictionary_data data/cc2vec/dataset_dict.pkl -save-dir model/cc2vec/snapshot/cc2vec/ftr  '

cc2ftr_predict_train = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_cc2ftr -batch_size 256 -predict -predict_data  data/cc2vec/features_train.pkl -dictionary_data  data/cc2vec/dataset_dict.pkl -load_model model/cc2vec/snapshot/cc2vec/ftr/cc2vec/epoch_50.pt -name  train_cc2ftr.pkl '

cc2ftr_predict_test = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_cc2ftr -batch_size 256 -predict -predict_data  data/cc2vec/features_test.pkl -dictionary_data  data/cc2vec/dataset_dict.pkl -load_model model/cc2vec/snapshot/cc2vec/ftr/cc2vec/epoch_50.pt -name  test_cc2ftr.pkl'

deepjit_train = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_DExtended -train -train_data data/deepjit/features_train.pkl -train_data_cc2ftr data/cc2vec/train_cc2ftr.pkl -dictionary_data data/deepjit/dataset_dict.pkl -save-dir model/cc2vec/snapshot/cc2vec/model'

deepjit_predict = "CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_DExtended -predict -pred_data data/deepjit/features_test.pkl -pred_data_cc2ftr data/cc2vec/test_cc2ftr.pkl -dictionary_data data/deepjit/dataset_dict.pkl -load_model model/cc2vec/snapshot/cc2vec/model/epoch_50.pt "


def CC2Vec_train_and_eval():
    visible_device = 0
    cmd1 = cc2ftr_train.format(visible_device)
    print(cmd1)
    print("<<<<<<<<<<<<<<<<<<<< Step 1: training cc2vec>>>>>>>>>>>>>>>>>>>")
    os.system(cmd1)

    cmd2 = cc2ftr_predict_train.format(visible_device)
    print(cmd2)
    print("<<<<<<<<<<<<<<<<<<<< Step 2: get cc2vec's representation for deepjit train data>>>>>>>>>>>>>>>>>>>")
    os.system(cmd2)

    cmd3 = cc2ftr_predict_test.format(visible_device)
    print(cmd3)
    print("<<<<<<<<<<<<<<<<<<<< Step 3: get cc2vec's representation for deepjit test data>>>>>>>>>>>>>>>>>>>")
    os.system(cmd3)

    cmd4 = deepjit_train.format(visible_device)
    print(cmd4)
    print("<<<<<<<<<<<<<<<<<<<< Step 4: training deepjit combined cc2vec representation>>>>>>>>>>>>>>>>>>>")
    os.system(cmd4)

    cmd5 = deepjit_predict.format(visible_device)
    print(cmd5)
    print("<<<<<<<<<<<<<<<<<<<< Step 5: evaluating model>>>>>>>>>>>>>>>>>>>")
    os.system(cmd5)


if __name__ == "__main__":
    print("Runing CC2Vec model")
    CC2Vec_train_and_eval()
