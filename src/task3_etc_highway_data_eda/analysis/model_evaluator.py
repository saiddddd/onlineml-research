import abc
import math
import copy
from tqdm import tqdm

from tools.model_perform_visualization import PredictionProbabilityDist, TrendPlot

from sklearn.metrics import accuracy_score, recall_score, f1_score

class ModelEvaluator(abc.ABC):

    def __init__(self, model, data_loader_for_testing, label):

        self._model = model
        self._data_loader_for_testing = data_loader_for_testing
        self._label = label

    @abc.abstractmethod
    def run_prediction_probability_distribution_checker(self):
        pass


class SklearnModelEvaluator(ModelEvaluator):
    def __init__(self, model, data_loader_for_testing, label):
        super(SklearnModelEvaluator, self).__init__(model, data_loader_for_testing, label)

    def run_prediction_probability_distribution_checker(self, save_plot_path=None):

        full_testing_dataset = self._data_loader_for_testing.get_full_df()

        X_test = copy.deepcopy(full_testing_dataset)
        y_test = X_test.pop(self._label)
        X_test.drop(["DateTime"], axis=1, inplace=True)

        pred_proba_result = self._model.predict_proba(X_test)
        draw_pred_proba = PredictionProbabilityDist(pred_proba_result, y_test)
        draw_pred_proba.draw_proba_dist_by_true_false_class_seperated()

        if save_plot_path is not None:
            draw_pred_proba.save_fig(save_plot_path)
            print("Saving prediction probability distribution plot successfully")


    def run_accuracy_and_recall_trend_checker(self, save_plot_path=None):
        num_target_list = []
        acc_trend_list = []
        recall_trend_list = []
        recall_uncertainty_list = []
        f1_score_list = []

        for i_date in self._data_loader_for_testing.get_distinct_date_set_list():
            sub_df_by_date = self._data_loader_for_testing.get_sub_df_by_date(i_date)
            X_test = copy.deepcopy(sub_df_by_date)
            y_test = X_test.pop(self._label)
            X_test.drop(["DateTime"], axis=1, inplace=True)

            pred_proba_result = self._model.predict_proba(X_test)

            pred_proba_casting_binary = list(map(lambda x: 0 if x < 0.4 else 1, pred_proba_result[:, 1]))
            num_target = y_test.tolist().count(1)

            acc = accuracy_score(y_test, pred_proba_casting_binary)
            recall = recall_score(y_test, pred_proba_casting_binary)
            recall_uncertainty = math.sqrt(recall * (1 - recall) / num_target)
            f1_s = f1_score(y_test, pred_proba_casting_binary, average='weighted')

            num_target_list.append(num_target)
            acc_trend_list.append(acc * 100)
            recall_trend_list.append(recall * 100)
            recall_uncertainty_list.append(recall_uncertainty * 100)
            f1_score_list.append(f1_s)

        x_list = self._data_loader_for_testing.get_distinct_date_set_list()
        trend_plot = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
        trend_plot.plot_trend(x_list, acc_trend_list, label="accuracy")
        trend_plot.plot_trend_with_error_bar(x_list, recall_trend_list, yerr=recall_uncertainty_list, markersize=4, capsize=2, label="recall rate")

        if save_plot_path is not None:
            trend_plot.save_fig(title="Acc Trend Plot", save_fig_path=save_plot_path)



class RiverModelEvaluator(ModelEvaluator):
    def __init__(self, model, data_loader_for_testing, label):
        super(RiverModelEvaluator, self).__init__(model, data_loader_for_testing, label)

    def run_prediction_probability_distribution_checker(self, save_plot_path=None):

        full_testing_dataset = self._data_loader_for_testing.get_full_df()

        X_test = copy.deepcopy(full_testing_dataset)
        y_test = X_test.pop(self._label)
        X_test.drop(["DateTime"], axis=1, inplace=True)

        pred_proba_result = []
        for index, raw in tqdm(X_test.iterrows(), total=X_test.shape[0]):
            pred_proba_result_element = self._model.predict_one(raw)
            pred_proba_result.append(pred_proba_result_element)

        draw_pred_proba = PredictionProbabilityDist(pred_proba_result, y_test)
        draw_pred_proba.draw_proba_dist_by_true_false_class_seperated()

        if save_plot_path is not None:
            draw_pred_proba.save_fig(save_plot_path)
            print("Saving prediction probability distribution plot successfully")


    def run_accuracy_and_recall_trend_checker(self, save_plot_path=None):
        num_target_list = []
        acc_trend_list = []
        recall_trend_list = []
        recall_uncertainty_list = []
        f1_score_list = []

        for i_date in self._data_loader_for_testing.get_distinct_date_set_list():
            sub_df_by_date = self._data_loader_for_testing.get_sub_df_by_date(i_date)
            X_test = copy.deepcopy(sub_df_by_date)
            y_test = X_test.pop(self._label)
            X_test.drop(["DateTime"], axis=1, inplace=True)

            pred_proba_result = []
            for index, raw in tqdm(X_test.iterrows(), total=X_test.shape[0]):
                pred_proba_result_element = self._model.predict_one(raw)
                pred_proba_result.append(pred_proba_result_element)

            pred_proba_casting_binary = list(map(lambda x: 0 if x < 0.4 else 1, pred_proba_result[:, 1]))
            num_target = y_test.tolist().count(1)

            acc = accuracy_score(y_test, pred_proba_casting_binary)
            recall = recall_score(y_test, pred_proba_casting_binary)
            recall_uncertainty = math.sqrt(recall * (1 - recall) / num_target)
            f1_s = f1_score(y_test, pred_proba_casting_binary, average='weighted')

            num_target_list.append(num_target)
            acc_trend_list.append(acc * 100)
            recall_trend_list.append(recall * 100)
            recall_uncertainty_list.append(recall_uncertainty * 100)
            f1_score_list.append(f1_s)

        x_list = self._data_loader_for_testing.get_distinct_date_set_list()
        trend_plot = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
        trend_plot.plot_trend(x_list, acc_trend_list, label="accuracy")
        trend_plot.plot_trend_with_error_bar(x_list, recall_trend_list, yerr=recall_uncertainty_list, markersize=4, capsize=2, label="recall rate")

        if save_plot_path is not None:
            trend_plot.save_fig(title="Acc Trend Plot", save_fig_path=save_plot_path)