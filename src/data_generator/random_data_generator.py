import time
from abc import ABC, abstractmethod

import random


class Generator(ABC):

    def __init__(self):
        self._generator_engine = None

    # def generate_one(self):


class Engine:

    def __init__(self, ):
        self._data_pattern = None

    # def generate_one(self):


class DataPatternEnsembler:

    def __init__(self):

        self._features = {}
        self._ensembler_function = {}

    def add_feature(self, feature_name: str, feature):
        if type(feature) is Features:
            self._features[feature_name] = feature
        else:
            print("Error, the feature: {} should in feature type".format(feature.__name__))

    # def set_ensemble_function(self,):
    #     self._ensembler_function =
    #
    # def get_ensemble_function(self):


class Features:

    def __init__(self):
        self.__ACCESSIBLE_FUNCTION = ['random', 'uniform', 'gauss', 'expovariate']
        self.__random = random

        self.__random_distribution_function = None
        self.__lower_bound = 0
        self.__upper_bound = 1
        self.__mu = 1
        self.__sigma = 1
        self.__lambda = 1


    @property
    def random_distribution_function(self):
        return self.__random_distribution_function

    @random_distribution_function.setter
    def random_distribution_function(self, func: str):
        if func not in self.__ACCESSIBLE_FUNCTION:
            print("Not Accessible function: {}! Please choose the one of following ".format(func))
            print( item + '\n' for item in self.__ACCESSIBLE_FUNCTION)
        self.__random_distribution_function = func

    @random_distribution_function.getter
    def random_distribution_function(self):
        return self.__random_distribution_function

    @property
    def lower_upper_bound(self):
        return self.__lower_bound, self.__upper_bound


    def get_feature_value(self):

        if self.random_distribution_function is None:
            raise RuntimeError

        feature_value = None

        if self.random_distribution_function == 'random':
            feature_value = self.__random.random()
        elif self.random_distribution_function == 'uniform':
            feature_value = self.__random.uniform(self.__lower_bound, self.__upper_bound)
        elif self.random_distribution_function == 'gauss':
            feature_value = self.__random.gauss(self.__mu, self.__sigma)
        elif self.random_distribution_function == 'expovariate':
            feature_value = self.__random.expovariate(self.__lambda)

        if feature_value is not None:
            return feature_value
        else:
            raise RuntimeError


if __name__ == '__main__':

    feature1 = Features()
    feature1.random_distribution_function = 'gauss'

    while True:
        value = feature1.get_feature_value()
        print(value)
        time.sleep(1)

