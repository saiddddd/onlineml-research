
from __future__ import annotations
from abc import ABC, abstractmethod
import copy

import traceback

import numpy as np
import pandas

from tools.DataPreparation import CreditCardPreparation, AirlineDataPreparation
from tools.DataVisualization import TrendPlot

from graphviz import Graph
import statistics

from river import tree
from river import preprocessing
from river import linear_model
from river import ensemble

from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

class ModelExperimentWorkflow(ABC):
    """
    An Abstraction Factory interface declares a set of methods that represent complete
    model experiment and evaluation working pipeline.
    Base on the specific purpose, in case, training and probing model performance workflow
    will be slightly different.
    Concrete class will be implement for specific purpose.
    """

    def __init__(self, input_data_preparation_class, random_seed=42):
        """
        Initializing Model Performacne testing workflow.
        """
        if isinstance(
                 input_data_preparation_class,
                (AirlineDataPreparation, CreditCardPreparation)):
            pass
        else:
           raise TypeError

        '''
            initialization of default dataset.
            prepare:
             1. full set(without split into training-testing dataset)
             2. split training-testing dataset, the dataset can be used in general model training process
        '''
        #=============================================#
        # Prepare the full dataset, without splitting #
        #=============================================#
        self._full_data_features, self._full_data_target = input_data_preparation_class.get_pd_df_data()
        #===========================#
        # Prepare the split dataset #
        #===========================#
        self._train_features, self._test_features, self._train_target, self._test_target = \
            input_data_preparation_class.get_splitted_train_test_pd_df_data(training_dataset_ratio=0.7, random_seed=random_seed)

        self._model_init = None
        self._model = None

        self._inc_tot_pred_result = []
        self._inc_tot_pred_related_target = []

        self._inc_tot_pred_cnt = 0
        self._inc_tot_correct_cnt = 0
        self._inc_tot_target_cnt = 0
        self._inc_tot_pred_target_cnt = 0
        self._inc_tot_pickup_target_cnt = 0
        self._inc_summary_acc = float

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def train_model_by_arbitrary_full_data(self):
        pass

    @abstractmethod
    def train_model_by_arbitrary_split_train_data(self):
        pass

    @abstractmethod
    def batch_pred_test_dataset(self) -> (list, list):
        """
        batch testing on split test dataset,
        :return: predict result and target ground true.
        """
        pass

    @abstractmethod
    def batch_pred_arbitrary_full_data(self, start_row, end_row) -> (list, list):
        """
        predict subset data restricted between start_row, and end_row
        return predict result and target ground true back to control flow for evaluation
        :param start_row: start point of row of data.
        :param end_row: end point of row of data.
        :return: predict result and target ground true.
        """
        pass

    @abstractmethod
    def incremental_prediction(self, start_row=int, end_row=int) -> None:
        """
        predict subset data restricted between start_row, and end_row
        accumulated predict result internally, will not return by this method
        :param start_row:
        :param end_row:
        :return: None
        """
        pass

    def batch_evaluate_model_get_accuracy_recall(self, pred_result=[], input_test_target=None):
        # Simply Prediction
        if type(input_test_target) is None:
            test_target = np.array(self._test_target)
        else:
            test_target = np.array(input_test_target)
        correct_cnt = 0
        target_cnt = 0
        pred_target_cnt = 0
        pickup_target_cnt = 0
        for i in range(len(test_target)):
            if test_target[i] == 1:
                target_cnt += 1
                if pred_result[i] == test_target[i]:
                    pickup_target_cnt += 1
            if pred_result[i] == 1:
                pred_target_cnt += 1
            if pred_result[i] == test_target[i]:
                correct_cnt += 1

        print("Acc: {:.2f}".format(correct_cnt / len(test_target) * 100))
        return correct_cnt / len(test_target), pickup_target_cnt / target_cnt

    def incremental_evaluate_model_get_accuracy_recall(self):
        # Simply Prediction
        test_target = np.array(self._inc_tot_pred_related_target)
        correct_cnt = 0
        target_cnt = 0
        pred_target_cnt = 0
        pickup_target_cnt = 0
        for i in range(len(test_target)):
            if test_target[i] == 1:
                target_cnt += 1
                if self._inc_tot_pred_result[i] == test_target[i]:
                    pickup_target_cnt += 1
            if self._inc_tot_pred_result[i] == 1:
                pred_target_cnt += 1
            if self._inc_tot_pred_result[i] == test_target[i]:
                correct_cnt += 1

        print("Acc: {:.2f}".format(correct_cnt / len(test_target) * 100))
        return correct_cnt / len(test_target), pickup_target_cnt / target_cnt

    def get_model_type(self):
        return type(self._model)

    def set_model(self, model):
        self._model_init = model
        self._model = copy.copy(self._model_init)

    def reset_model(self):
        self._model = copy.copy(self._model_init)

    def limited_rows_full_dataset(self, limit_num=100) -> pandas.DataFrame:
        """
        return limited rows of training dataset for specific use.
        Base on Full Dataset self._full_data_feature and self._full_data_target
        :param limit_num: default 100
        :return: limited_row_feature, limited_row_target
        """
        if type(limit_num) is not int:
            raise TypeError
        else:
            limited_row_features = self._full_data_features.head(n=limit_num)
            limited_row_target = self._full_data_target.head(n=limit_num)
            return limited_row_features, limited_row_target

    def subset_rows_arbitrary_full_dataset(self, start_row=int, end_row=int) -> pandas.DataFrame:
        """
        return arbitrary subset based on full dataset, restricted between start_row, and end_row
        :param start_row:
        :param end_row:
        :return:
        """

        subset_row_features = self._full_data_features[start_row:end_row]
        subset_row_target = self._full_data_target[start_row:end_row]
        return subset_row_features, subset_row_target

    def limited_rows_split_train_dataset(self, limit_num=100) -> pandas.DataFrame:
        """
        return limited rows of training dataset for specific use.
        Base on Split Dataset self._train_features and self._train_target
        :param limit_num:
        :return:
        """
        if type(limit_num) is not int:
            raise TypeError
        else:
            limited_row_features = self._train_features.head(n=limit_num)
            limited_row_target = self._train_target.head(n=limit_num)
            return limited_row_features, limited_row_target

    def subset_rows_arbitrary_split_train_dataset(self, start_row=int, end_row=int) -> pandas.DataFrame:
        """
        return arbitrary subset based on split training dataset, restricted between start_row, and end_row
        :param start_row:
        :param end_row:
        :return:
        """

        subset_row_features = self._train_features[start_row:end_row]
        subset_row_target = self._train_target[start_row:end_row]
        return subset_row_features, subset_row_target



class ModelSklearnWorkflow(ModelExperimentWorkflow):

    def train_model(self, train_num_limit=int):

        data_for_train, target_for_train = self.limited_rows_split_train_dataset(train_num_limit)
        print('Training Features Shape: ', data_for_train.shape)
        print('Training Target Shape: ', target_for_train.shape)

        self._model.fit(data_for_train, target_for_train)

    def train_model_by_arbitrary_full_data(self, sub_data_start_row, sub_data_end_row):

        data_for_train, target_for_train = self.subset_rows_arbitrary_full_dataset(
            start_row=sub_data_start_row,
            end_row=sub_data_end_row
        )
        self._model.fit(data_for_train, target_for_train)

    def train_model_by_arbitrary_split_train_data(self, sub_data_start_row, sub_data_end_row):

        data_for_train, target_for_train = self.subset_rows_arbitrary_split_train_dataset(
            start_row=sub_data_start_row,
            end_row=sub_data_end_row
        )
        self._model.fit(data_for_train, target_for_train)

    def batch_pred_test_dataset(self):
        pred_result = self._model.predict(self._test_features)
        return pred_result, self._test_target

    def batch_pred_arbitrary_full_data(self, sub_data_start_row, sub_data_end_row) -> (list, list):

        data_for_test, target_for_test = self.subset_rows_arbitrary_full_dataset(
            start_row=sub_data_start_row,
            end_row=sub_data_end_row
        )

        pred_result = self._model.predict(data_for_test)
        return pred_result, target_for_test

    def incremental_prediction(self, start_row=int, end_row=int) -> None:

        data_for_test, target_for_test = self.subset_rows_arbitrary_full_dataset(
            start_row=start_row,
            end_row=end_row
        )

        pred_result = self._model.predict(data_for_test)
        self._inc_tot_pred_result.extend(pred_result)
        self._inc_tot_pred_related_target.extend(target_for_test)




class ModelRiverOnlineMLWorkflow(ModelExperimentWorkflow):

    def train_model(self, is_reset_mode=True, train_num_limit=int):
        """
        training model method on RiverML online learning implementation has slightly different from sklearn
        here have to check model is going to do incremental learning or reset model (re-train)
        :param is_reset_mode: will this training process reset the model(not incremental)? Default=True
        :param train_num_limit:
        :return:
        """
        data_for_train, target_for_train = self.limited_rows_split_train_dataset(train_num_limit)
        print('Training Features Shape: ', data_for_train.shape)
        print('Training Target Shape: ', target_for_train.shape)

        #-------------------------------------------------#
        # Using River ML model to do online training      #
        # Core section, the different from SK learn model #
        #-------------------------------------------------#
        if is_reset_mode:
            # reset model(not incremental from previous step)
            self.reset_model()

        for index, raw in tqdm(data_for_train.iterrows(), total=data_for_train.shape[0]):
            self._model.learn_one(raw, target_for_train[index])

    def train_model_by_arbitrary_full_data(self, sub_data_start_row, sub_data_end_raw, is_reset_mode=True):

        data_for_train, target_for_train = self.subset_rows_arbitrary_full_dataset(
            start_row=sub_data_start_row,
            end_row=sub_data_end_raw
        )
        print('Training Features Shape: ', data_for_train.shape)
        print('Training Target Shape: ', target_for_train.shape)

        # -------------------------------------------------#
        # Using River ML model to do online training      #
        # Core section, the different from SK learn model #
        # -------------------------------------------------#
        if is_reset_mode:
            # reset model(not incremental from previous step)
            self.reset_model()

        for index, raw in tqdm(data_for_train.iterrows(), total=data_for_train.shape[0]):
            self._model.learn_one(raw, target_for_train[index])

    def train_model_by_arbitrary_split_train_data(self, sub_data_start_row, sub_data_end_raw, is_reset_mode=True):

        data_for_train, target_for_train = self.subset_rows_arbitrary_split_train_dataset(
            start_row=sub_data_start_row,
            end_row=sub_data_end_raw
        )
        print('Training Features Shape: ', data_for_train.shape)
        print('Training Target Shape: ', target_for_train.shape)

        # -------------------------------------------------#
        # Using River ML model to do online training      #
        # Core section, the different from SK learn model #
        # -------------------------------------------------#
        if is_reset_mode:
            # reset model(not incremental from previous step)
            self.reset_model()

        for index, raw in tqdm(data_for_train.iterrows(), total=data_for_train.shape[0]):
            self._model.learn_one(raw, target_for_train[index])


    def batch_pred_test_dataset(self):
        pred_result = []
        for index, raw in tqdm(self._test_features.iterrows(), total=self._test_features.shape[0]):
            test_pred = self._model.predict_one(raw)
            pred_result.append(test_pred)

        return pred_result, self._test_target

    def batch_pred_arbitrary_full_data(self, sub_data_start_row, sub_data_end_row) -> (list, list):

        pred_result = []
        data_for_test, target_for_test = self.subset_rows_arbitrary_full_dataset(
            start_row=sub_data_start_row,
            end_row=sub_data_end_row
        )

        for index, raw in tqdm(data_for_test.iterrows(), total=data_for_test.shape[0]):
            test_pred = self._model.predict_one(raw)
            pred_result.append(test_pred)

        return pred_result, target_for_test

    def incremental_prediction(self, start_row=int, end_row=int) -> None:

        pred_result = []
        data_for_test, target_for_test = self.subset_rows_arbitrary_full_dataset(
            start_row=start_row,
            end_row=end_row
        )

        for index, raw in tqdm(data_for_test.iterrows(), total=data_for_test.shape[0]):
            test_pred = self._model.predict_one(raw)
            pred_result.append(test_pred)

        self._inc_tot_pred_result.extend(pred_result)
        self._inc_tot_pred_related_target.extend(target_for_test)


if __name__ == '__main__':

    model_sklearn = RandomForestClassifier(
        n_estimators=50,
        criterion="entropy",
        max_depth=10,
        random_state=0
    )
    model_riverml_HTC = tree.HoeffdingTreeClassifier()
    model_riverml_AdaRF = ensemble.AdaptiveRandomForestClassifier()

    exp_flow_sklearn = ModelSklearnWorkflow(AirlineDataPreparation())
    exp_flow_riverml_HTC = ModelRiverOnlineMLWorkflow(AirlineDataPreparation())
    exp_flow_riverml_AdaRF = ModelRiverOnlineMLWorkflow(AirlineDataPreparation())

    exp_flow_sklearn.set_model(model_sklearn)
    exp_flow_riverml_HTC.set_model(model_riverml_HTC)
    exp_flow_riverml_AdaRF.set_model(model_riverml_AdaRF)

    exp_flow_sklearn.train_model_by_arbitrary_full_data(1, 501)
    exp_flow_riverml_HTC.train_model_by_arbitrary_full_data(1, 501)
    exp_flow_riverml_AdaRF.train_model_by_arbitrary_full_data(1, 501)

    exp_flow_sklearn.incremental_prediction(501, 2001)
    exp_flow_riverml_HTC.incremental_prediction(501, 2001)
    exp_flow_riverml_AdaRF.incremental_prediction(501, 2001)
    i_start = 2001
    i_size = 100
    i_step = 100

    x_text_point = []
    acc_result_sklearn = []
    acc_result_riverml_HTC = []
    acc_result_riverml_AdaRF = []

    while(True):

        x_text_point.append(i_start + 0.5 * i_size)

        # sklearn one step operation
        exp_flow_sklearn.incremental_prediction(i_start, i_start + i_size)
        acc, recall = exp_flow_sklearn.incremental_evaluate_model_get_accuracy_recall()
        acc_result_sklearn.append(acc)

        # riverml one step operation
        exp_flow_riverml_HTC.incremental_prediction(i_start, i_start + i_size)
        acc, recall = exp_flow_riverml_HTC.incremental_evaluate_model_get_accuracy_recall()
        acc_result_riverml_HTC.append(acc)
        exp_flow_riverml_HTC.train_model_by_arbitrary_full_data(i_start, i_start + i_size, is_reset_mode=False)

        #riverml adaptive random forest one step operation
        exp_flow_riverml_AdaRF.incremental_prediction(i_start, i_start + i_size)
        acc, recall = exp_flow_riverml_AdaRF.incremental_evaluate_model_get_accuracy_recall()
        acc_result_riverml_AdaRF.append(acc)
        exp_flow_riverml_AdaRF.train_model_by_arbitrary_full_data(i_start, i_start + i_size, is_reset_mode=False)

        i_start+=i_step

        if i_start > 70000:
            break

    x_text_point = [i * 0.0001 for i in x_text_point]
    acc_result_sklearn = [i * 100 for i in acc_result_sklearn]
    acc_result_riverml_HTC = [i * 100 for i in acc_result_riverml_HTC]
    acc_result_riverml_AdaRF = [i * 100 for i in acc_result_riverml_AdaRF]

    # print(x_text_point)
    # print(acc_result_sklearn)

    from tools.DataVisualization import TrendPlot

    aaa = TrendPlot()
    aaa.plot_trend(x_text_point, acc_result_sklearn, label='scikit learn RF')
    aaa.plot_trend(x_text_point, acc_result_riverml_HTC, label='HT classifier')
    aaa.plot_trend(x_text_point, acc_result_riverml_AdaRF, label='Adaptive RF')
    aaa.save_fig(
        title='Trend plot of Incremental ML model performance',
        x_label='#data accumulated x10000',
        y_label='accuracy (%)'
    )



    # model_river_online_ml = tree.HoeffdingTreeClassifier()
    # river_online_ml_process = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=43)
    # river_online_ml_process.set_model(model_river_online_ml)
    #
    # river_online_ml_process.train_model(is_reset_mode=True, train_num_limit=-1)
    # pred_result = river_online_ml_process.batch_pred_test_dataset()
    # acc, recall = river_online_ml_process.evaluate_model_get_accuracy_recall(pred_result=pred_result)

