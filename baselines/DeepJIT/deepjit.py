import os


def DeepJIT_train_and_eval(baseline_name: str = None):
    raw_train = 'python3 -m baselines.DeepJIT.main -train -train_data data/deepjit/features_train.pkl -dictionary_data datat/deepjit/dataset_dict.pkl  -device 0 -save_dir model/deepjit'

    raw_test = 'python3 -m baselines.DeepJIT.main -predict -pred_data data/deepjit/features_test.pkl -dictionary_data data/deepjit/dataset_dict.pkl -load_model model/deepjit/epoch_25.pt  -device 0'

    # train
    print("**********************Training*********************")
    os.system(raw_train)

    print("**********************Evaluating*********************")
    os.system(raw_test)


if __name__ == "__main__":
    print("Running deepjit model")
    DeepJIT_train_and_eval('deepjit')
    # pass
