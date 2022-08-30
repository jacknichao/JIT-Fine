from baselines.CC2Vec.jit_DExtended_model import DeepJITExtended
from baselines.CC2Vec.jit_utils import mini_batches_DExtended
import torch
from tqdm import tqdm
import pandas as pd
import os


from baselines.utils.performance_measure import PerformanceMeasure
from baselines.utils.results_writer import ResultWriter
from baselines.utils.preprocess_data import load_deepjit_test_dataframe




def evaluation_model(data, params):
    ids, cc2ftr, pad_msg, pad_code, labels, dict_msg, dict_code = data

    batches = mini_batches_DExtended(ids=ids, X_ftr=cc2ftr, X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]
    params.embedding_ftr = cc2ftr.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_ids, all_predict_prob, all_label = list(), list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            _ids, ftr, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict_prob = model.forward(ftr, pad_msg, pad_code)
                predict_prob = predict_prob.cpu().detach().numpy().tolist()
            else:
                predict_prob = model.forward(ftr, pad_msg, pad_code)
                predict_prob = predict_prob.detach().numpy().tolist()
            all_predict_prob += predict_prob
            all_label += labels.tolist()
            all_ids += _ids.tolist()


    # y_pred = np.round(all_predict)
    # print('y_pred is', y_pred)
    # print(list(all_predict))
    # auc_score = roc_auc_score(y_true=all_label, y_score=all_predict_prob)
    # print('Test data -- AUC score:', auc_score)

    print(len(all_ids), len(all_label), len(all_predict_prob))
    result_df = pd.DataFrame()
    result_df['commit_id'] = all_ids
    result_df['defective_commit_pred'] = [1 if p >= 0.5 else 0 for p in all_predict_prob]
    result_df['defective_commit_prob'] = all_predict_prob
    result_df = load_deepjit_test_dataframe(result_df)

    # auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict_prob)
    # print('Test data -- AUC score:', auc_score)

    presults = PerformanceMeasure().eval_metrics(result_df=result_df)
    print(presults)
    result_path = os.path.dirname(os.path.dirname(__file__)) + '/results/'
    ResultWriter().write_result(result_path=result_path, method_name="CC2Vec", presults=presults)
    return presults


