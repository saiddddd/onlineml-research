
from __future__ import annotations
from abc import ABC, abstractmethod

import traceback

import numpy as np
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

        self._train_features, self._test_features, self._train_target, self._test_target = \
            input_data_preparation_class.get_splitted_train_test_pd_df_data(training_dataset_ratio=0.7, random_seed=random_seed)

        self._model = None

    @abstractmethod
    def train_sklearn_model(self):
        pass


class ModelBatchPerformanceWorkflow(ModelExperimentWorkflow):

    def train_sklearn_model(self, train_num_limit=None):
        #=============================================#
        # Convert pandas to nparray                   #
        # For fit scikit learn model training require #
        #=============================================#
        train_features_npa = np.array(self._train_features)
        train_target_npa = np.array(self._train_target)
        test_features_npa = np.array(self._test_features)
        test_target_npa = np.array(self._test_target)
        if type(train_num_limit) is int:
            train_features_npa.head(n=train_num_limit)
            train_target_npa.head(n=train_num_limit)

        print('Training Features Shape: ', train_features_npa.shape)
        print('Training Target Shape: ', train_target_npa.shape)
        print('Testing Features Shape: ', test_features_npa.shape)
        print('Testing Target Shape: ', test_target_npa.shape)

        self._model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=0
        )





if __name__ == '__main__':
    ModelBatchPerformanceWorkflow()