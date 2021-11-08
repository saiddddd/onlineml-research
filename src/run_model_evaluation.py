
import argparse

import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score

from sklearn.ensemble import RandomForestClassifier
from river import tree, ensemble

from tools.DataPreparation import ArbitraryDataPreparation, CreditCardPreparation
from tools.DataVisualization import TrendPlot

from task1_model_evaluation.ModelExperimentWorkflow import ModelSklearnWorkflow, ModelRiverOnlineMLWorkflow


def get_data(input_data, label_name):

    if input_data == "creditcard":
        loaded_data = CreditCardPreparation()

    else:
        loaded_data = ArbitraryDataPreparation(
            input_data,
            label_name
        )
    return loaded_data


def init_workflow(args, loaded_data):
    x_df, y_df = loaded_data.get_pd_df_data()

    # --------------------------------#
    # Model and workflow preparation #
    # -------------------------------#
    exp_flow = None
    if args.model_type == 'sklearn':
        model_sklearn = RandomForestClassifier(
            n_estimators=50,
            criterion="entropy",
            max_depth=10,
            random_state=42
        )
        exp_flow = ModelSklearnWorkflow(
            loaded_data,
            random_seed=42
        )
        exp_flow.set_model(
            model_sklearn
        )
    elif args.model_type == 'river_htc':
        model_river_htc = tree.HoeffdingTreeClassifier()
        exp_flow = ModelRiverOnlineMLWorkflow(
            loaded_data,
            random_seed=42
        )
        exp_flow.set_model(model_river_htc)
    elif args.model_type == 'river_adarf':
        model_river_adarf = ensemble.AdaptiveRandomForestClassifier(
            n_models=50,
            max_depth=10,
            seed=0,
        )
        exp_flow = ModelRiverOnlineMLWorkflow(
            loaded_data,
            random_seed=42
        )
        exp_flow.set_model(model_river_adarf)

    else:
        print("choose a model type, sklearn, river_htc, river_adarf")

    return exp_flow

def sliding_proba_cut_threshold_draw(exp_flow, output_plot):
    proba_cut = []
    acc_slide_proba_cut = []
    recall_slide_proba_cut = []
    precision_slide_proba_cut = []

    for proba_threshold in np.arange(0.05, 0.65, 0.05):
        proba_cut.append(proba_threshold)
        print("prediction probability")
        pred_proba, true_y = exp_flow.batch_pred_prob_test_dataset()  # used prediction probability
        pred_proba_cast = list(map(lambda x: 0 if x < proba_threshold else 1,
                                   pred_proba[:, 1]))  # casting probability to final true/false classify

        acc = accuracy_score(true_y, pred_proba_cast)
        recall = recall_score(true_y, pred_proba_cast)
        precision = precision_score(true_y, pred_proba_cast)
        acc_slide_proba_cut.append(acc)
        recall_slide_proba_cut.append(recall)
        precision_slide_proba_cut.append(precision)
        # print("acc, recall, precision extracted by prediction probability casting threshold {}".format(proba_threshold))
        # print(acc, recall, precision)

    plot = TrendPlot()
    plot.plot_trend(proba_cut, acc_slide_proba_cut, label="acc")
    plot.plot_trend(proba_cut, recall_slide_proba_cut, label="recall rate")
    plot.plot_trend(proba_cut, precision_slide_proba_cut, label='precision')
    plot.save_fig("sliding proba cut point", x_label="proba cut", y_label="index", save_fig_path=output_plot)



def run_complete_flow(args, loaded_data):

    exp_flow = init_workflow(args, loaded_data)
    # training model

    print("Training Model")
    exp_flow.train_model()
    print("Training Model Complete")

    print("prediction")
    pred_y, true_y =exp_flow.batch_pred_test_dataset()  # using prediction

    acc = accuracy_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    print("acc, recall, precision extracted by prediction value")
    print(acc, recall, precision)

    sliding_proba_cut_threshold_draw(exp_flow, args.output_plot)


def run_incremental_recall_proba_cut(args, loaded_data, output_plot_stash):

    exp_flow = init_workflow(args, loaded_data)

    exp_flow.train_model_by_arbitrary_split_train_data(1, 501)
    exp_flow.incremental_prediction_proba(501, 2001)

    x_test_point = []
    evaluation_result = []

    i_start = 2001
    i_size = 100
    i_step = 100
    while (True):

        x_test_point.append(i_start + 0.5 * i_size)

        if args.predict_type == 'pred_proba':
            pred_proba_cut_threshold = args.proba_threshold
            exp_flow.incremental_prediction_proba(i_start, i_start + i_size, pred_proba_cut_threshold)
        elif args.predict_type == 'pred':
            exp_flow.incremental_prediction(i_start, i_start + i_size)
        else:
            exp_flow.incremental_prediction(i_start, i_start + i_size)
            print("Warning, please check prediction type, is pred or pred_proba? Using pred this time")


        acc, recall = exp_flow.incremental_evaluate_model_get_accuracy_recall()

        if args.evaluation_index == "recall":
            evaluation_result.append(recall)
        elif args.evaluation_index == "accuracy":
            evaluation_result.append(acc)


        if "river" in args.model_type:
            exp_flow.train_model_by_arbitrary_split_train_data(i_start, i_start + i_size, is_reset_mode=False)

        i_start += i_step

        if i_start > 70000:
            break


    output_plot_stash.plot_trend(x_test_point, evaluation_result, label=args.evaluation_index)
    # plot.save_fig("sliding proba cut point", x_label="proba cut", y_label="*100%", save_fig_path=args.output_plot)




def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-data', type=str)
    parser.add_argument('-l', '--label-name', type=str)
    parser.add_argument('-m', '--model-type', type=str, default='river_htc')
    parser.add_argument('-e', '--evaluation-index', type=str, default='recall')
    parser.add_argument('-p', '--predict-type', type=str, default='pred', help='prediction type, 1. pred or 2. pred_proba')
    parser.add_argument('-t', '--proba-threshold', type=float, default=0.2, help='pred_proba cut threshold')
    parser.add_argument('-o', '--output-plot', type=str, default='./output_plot/test.pdf')
    args = parser.parse_args()

    loaded_data = get_data(
        args.input_data,
        args.label_name
    )

    plot = TrendPlot()
    # run_complete_flow(args, loaded_data)
    run_incremental_recall_proba_cut(args, loaded_data, plot)
    plot.save_fig("sliding proba cut point", x_label="proba cut", y_label="*100%",
                  save_fig_path=args.model_type+'_'+args.predict_type+'_'+str(args.proba_threshold)+'_'+args.output_plot
                  )

if __name__ == '__main__':
    run_main()
