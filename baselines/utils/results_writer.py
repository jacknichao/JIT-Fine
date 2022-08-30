import os


class ResultWriter:

    def write_result(self, result_path, method_name, presults: dict):
        result_path = result_path + 'result.csv'
        if not os.path.exists(result_path):
            file = open(result_path, "a+")
            file.write(
                "method,f1,auc,recall_at_20_percent_effort,effort_at_20_percent_LOC_recall,p_opt\n")
            file.flush()

        with open(result_path, "a+") as file:
            file.write(
                "{method_name},{f1},{auc},{recall_at_20_percent_effort},{effort_at_20_percent_LOC_recall},{p_opt}\n".format(
                    method_name=method_name,
                    f1=presults["f1"], auc=presults["auc"],
                    recall_at_20_percent_effort=presults["recall_at_20_percent_effort"],
                    effort_at_20_percent_LOC_recall=presults["effort_at_20_percent_LOC_recall"],
                    p_opt=presults["p_opt"]))

            file.flush()
        print("{}result saved successfully".format(method_name))
