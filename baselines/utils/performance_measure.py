from sklearn.metrics import *
import pandas as pd
import math
import numpy as np


class PerformanceMeasure:
    """
    calculate effort-aware and effort-free metrics
    """

    def get_recall_at_k_percent_effort(self, percent_effort, result_df_arg, real_buggy_commits):
        cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
        buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
        buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1.0]
        recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

        return recall_k_percent_effort

    def eval_metrics(self, result_df: pd.DataFrame):
        pred = result_df['defective_commit_pred']
        y_test = result_df['label']

        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5

        AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

        result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
        result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

        result_df = result_df.sort_values(by='defect_density', ascending=False)
        actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
        actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

        result_df['cum_LOC'] = result_df['LOC'].cumsum()
        actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
        actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

        real_buggy_commits = result_df[result_df['label'] == 1.0]

        label_list = list(result_df['label'])

        all_rows = len(label_list)

        # find Recall@20%Effort
        cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
        buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
        buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1.0]
        recall_at_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

        # find Effort@20%Recall
        buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
        buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

        # find P_opt
        percent_effort_list = []
        predicted_recall_at_percent_effort_list = []
        actual_recall_at_percent_effort_list = []
        actual_worst_recall_at_percent_effort_list = []

        for percent_effort in np.arange(10, 101, 10):
            predicted_recall_k_percent_effort = self.get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                                    real_buggy_commits)
            actual_recall_k_percent_effort = self.get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                                 real_buggy_commits)
            actual_worst_recall_k_percent_effort = self.get_recall_at_k_percent_effort(percent_effort,
                                                                                       actual_worst_result_df,
                                                                                       real_buggy_commits)

            percent_effort_list.append(percent_effort / 100)

            predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
            actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
            actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

        p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                      auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                     (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                      auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

        return {
                "f1": round(f1, 4),
                "auc": round(AUC, 4),
                "recall_at_20_percent_effort": round(recall_at_20_percent_effort, 4),
                "effort_at_20_percent_LOC_recall": round(effort_at_20_percent_LOC_recall, 4),
                "p_opt": round(p_opt, 4)
                }

