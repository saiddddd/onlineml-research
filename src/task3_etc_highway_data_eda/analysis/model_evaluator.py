import abc
import math
import copy
import numpy as np
from tqdm import tqdm

from tools.model_perform_visualization import PredictionProbabilityDist, TrendPlot

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import RocCurveDisplay

class ModelEvaluator(abc.ABC):

    def __init__(self, model, data_loader_for_testing, label):

        self._model = model
        self._data_loader_for_testing = data_loader_for_testing
        self._label = label


    def run_prediction_probability_distribution_checker(self, save_plot_path=None):
        # =======================================================================#
        # Running prediction probability and get full set of prediction result. #
        # =======================================================================#
        pred_proba_result, y_test = self.predict_proba_true_class_full_set()

        draw_pred_proba = PredictionProbabilityDist(pred_proba_result, y_test)
        draw_pred_proba.draw_proba_dist_by_true_false_class_seperated()

        if save_plot_path is not None:
            draw_pred_proba.save_fig(save_plot_path)
            print("Saving prediction probability distribution plot successfully")

    @abc.abstractmethod
    def predict_proba_true_class_full_set(self) -> list:
        """
        run prediction probability through full dataset, return prediction probability result and corresponding answer
        :return:
        """
        pass

    @abc.abstractmethod
    def predict_proba_true_class_by_date(self, i_date=None):
        pass

    def get_testing_x_y_full_set(self):

        full_testing_dataset = self._data_loader_for_testing.get_full_df()

        X_test = copy.deepcopy(full_testing_dataset)
        y_test = X_test.pop(self._label)
        X_test.drop(["DateTime"], axis=1, inplace=True)

        return X_test, y_test

    def get_testing_x_y_daily_subset(self, i_date):

        sub_df_by_date = self._data_loader_for_testing.get_sub_df_by_date(i_date)

        X_test = copy.deepcopy(sub_df_by_date)
        y_test = X_test.pop(self._label)
        X_test.drop(["DateTime"], axis=1, inplace=True)

        return X_test, y_test

    @staticmethod
    def get_model_score_by_daily_subset(pred_proba_result, y_test, proba_cut=0.5):
        """
        casting probability as binary class 1 or 0.
        provided proba_cut threshold for discriminate 1 or 0.
        default cut point set as 0.5, binary class is 1 if probability > 0.5 else 0
        After casting probability prediction into binary, do model performance scoring operation
        return accuracy, recall, recall uncertainty and f1 score
        :param pred_proba_result: prediction probability list going to casting
        :param y_test: target list
        :param proba_cut: probability casting cut point(threshold)
        :return: accuracy, recall, recall_uncertainty, f1_score
        """
        try:
            pred_proba_casting_binary = list(map(lambda x: 0 if x < proba_cut else 1, pred_proba_result))
            if isinstance(y_test, list):
                num_target = y_test.count(1)
            else:
                num_target = y_test.tolist().count(1)

            acc = accuracy_score(y_test, pred_proba_casting_binary)
            recall = recall_score(y_test, pred_proba_casting_binary, zero_division=1)
            recall_uncertainty = math.sqrt(recall * (1 - recall) / num_target)
            recall_uncertainty = 0
            f1_s = f1_score(y_test, pred_proba_casting_binary, zero_division=1)

            # calculating auc
            auc_score = roc_auc_score(y_test, pred_proba_result, labels=1)

        except ValueError:
            auc_score = 1

        return acc, recall, recall_uncertainty, f1_s, auc_score



    @staticmethod
    def roc_curve_displayer(pred_proba_result, y_test, estimator_name='estimator name'):
        """
        providing prediction probability result and y_true to draw roc
        :param pred_proba_result:
        :param y_test:
        :param estimator_name:
        :return: roc_curve_display
        """
        fpr, tpr, threshold = roc_curve(y_test, pred_proba_result, pos_label=1)
        cut_point_index = np.argmax(tpr - fpr)
        print("best threshold:{}  tpr:{}  fpr:{}".format(threshold[cut_point_index], tpr[cut_point_index], fpr[cut_point_index]))
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name)
        return display


class SklearnModelEvaluator(ModelEvaluator):
    def __init__(self, model, data_loader_for_testing, label):
        super(SklearnModelEvaluator, self).__init__(model, data_loader_for_testing, label)
    #
    # def run_prediction_probability_distribution_checker(self, save_plot_path=None):
    #

    def predict_proba_true_class_full_set(self):
        X_test, y_test = self.get_testing_x_y_full_set()
        pred_proba_result = self._model.predict_proba(X_test)
        return pred_proba_result[:, 1], y_test

        
    def predict_proba_true_class_by_date(self, i_date=None):
        # sub_df_by_date = self._data_loader_for_testing.get_sub_df_by_date(i_date)
        # X_test = copy.deepcopy(sub_df_by_date)
        # y_test = X_test.pop(self._label)
        # X_test.drop(["DateTime"], axis=1, inplace=True)

        if i_date is None:
            X_test, y_test = self.get_testing_x_y_full_set()
        else:
            X_test, y_test = self.get_testing_x_y_daily_subset(i_date)

        pred_proba_result = self._model.predict_proba(X_test)

        return pred_proba_result[:, 1], y_test



class RiverModelEvaluator(ModelEvaluator):
    def __init__(self, model, data_loader_for_testing, label):
        super(RiverModelEvaluator, self).__init__(model, data_loader_for_testing, label)


    def predict_proba_true_class_full_set(self, i_date=None):
        X_test, y_test = self.get_testing_x_y_full_set()
        pred_proba_result_list = []
        for index, raw in tqdm(X_test.iterrows(), total=X_test.shape[0]):
            try:
                pred_proba_result = self._model.predict_proba_one(raw)
                if isinstance(pred_proba_result, dict):
                    pred_proba_result_list.append(pred_proba_result.get(1))
                
            except:
                print("error happen")
        
        return np.array(pred_proba_result_list), y_test
            
    def predict_proba_true_class_by_date(self, i_date=None, do_online_training=False):

        # sub_df_by_date = self._data_loader_for_testing.get_sub_df_by_date(i_date)
        # X_test = copy.deepcopy(sub_df_by_date)
        # y_test = X_test.pop(self._label)
        # X_test.drop(["DateTime"], axis=1, inplace=True)

        if i_date is None:
            X_test, y_test = self.get_testing_x_y_full_set()
        else:
            X_test, y_test = self.get_testing_x_y_daily_subset(i_date)

        pred_proba_result_list = []
        corresponding_ans = []
        for index, raw in tqdm(X_test.iterrows(), total=X_test.shape[0]):
            try:
                pred_proba_result = self._model.predict_proba_one(raw)
                if isinstance(pred_proba_result, dict):
                    pred_proba_result_element = pred_proba_result.get(1)
                    if pred_proba_result_element is not None:
                        pred_proba_result_list.append(pred_proba_result.get(1))
                        corresponding_ans.append(y_test[index])
                    else:
                        print('Critical error: prediction proba result from riverml Model is None')
                else:
                    print('Critical error: prediction proba result from riverml Model is not dict type (unexpect type found !) ')

            except:
                print("error happen while prediction")

            if do_online_training:
                '''
                if do online training option is on, go to do it, or skip this part
                '''
                try:
                    #-------------------------------------------------#
                    # if prediction is fine, go learn one observation #
                    #-------------------------------------------------#
                    self._model.learn_one(raw, y_test[index])
                except:
                    print("error happen while learning")
                
        if all(isinstance(x, float) for x in pred_proba_result_list):
            return pred_proba_result_list, corresponding_ans
        else:
            print("river model prediction probabality cause error, not all return result provide float type!!, check")
            raise RuntimeError
