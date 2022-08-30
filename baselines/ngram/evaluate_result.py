from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
import os


def get_line_level_metrics(line_score, label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))  # cannot pass line_score as list T-T
    pred = np.round(line_score)

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0
    top_5_acc = 0
    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))


    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 & top-5 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])
        if all_rows < 5:
            top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

        # find recall
        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        # find effort @20% LOC recall
        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        assert int(buggy_20_percent_row_num) <= float(len(line_df))
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def eval_result(data, jitfine_result):
    result_file = './line-level-result-onlyadds.txt'
    # load jitfine and ngram results respectively, only evaluate on the instances that pred=1 & label = 1 by jitfine
    result_df = pd.read_csv(result_file, sep='\t')
    jitfine_result = pd.read_csv(jitfine_result, sep='\t')
    jitfine_result.columns = ['commit_id', 'prob', 'pred', 'label']
    result_df.columns = ['commit_id', 'line_idx', 'score', 'label']

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    commits = result_df['commit_id'].unique()
    for commit_id in commits:
        cur_jitfine = jitfine_result[
            (jitfine_result['commit_id'] == commit_id) & (jitfine_result['pred'] == 1) & (jitfine_result['label'] == 1)]
        if cur_jitfine.empty:
            continue
        cur_result = result_df[result_df['commit_id'] == commit_id]
        cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = get_line_level_metrics(
            cur_result['score'].tolist(), cur_result['label'].tolist())
        IFA.append(cur_IFA)
        top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
        effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
        top_10_acc.append(cur_top_10_acc)
        top_5_acc.append(cur_top_5_acc)

    print(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )


if __name__ == '__main__':
    file = 'data/ngram/changes_complete_buggy_line_level.pkl'
    data = pd.read_pickle(file)
    # get prediction file
    jitfine_output_dir = 'model/jitfine/saved_models_concat/'
    jitfine_result = os.path.join(jitfine_output_dir, f'predictions.csv')
    eval_result(data, jitfine_result)
