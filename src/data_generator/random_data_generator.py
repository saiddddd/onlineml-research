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

    def __str__(self):
        return str(self._features)

    def add_feature(self, feature_name: str, feature):
        if type(feature) is Features:
            self._features[feature_name] = feature
        else:
            print("Error, the feature: {} should in feature type".format(feature.__name__))

    def set_ensemble_function(self, pattern_name: str, involve_features: list, coefficient: list, operator: list):
        pass

        #y = x1 + 2x2 - x3 / x4
        #involve_features = [x1, x2, x3, x4]
        #coefficient = [1, 2, 1, 1]
        #operator = [+, - , /]
        # x = involve_features[i]*coefficient[i]
        # xx = xx operator x

        # self._ensembler_function =

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
        self.__exp_lambda = 1


    @property
    def random_distribution_function(self):
        """
        property for setting distribution function
        there is `random`, `uniform`, `gauss`, and `expovariate` distribution functions can be set.
        basically, those function provided by `random()` can be used.
        :return:
        """
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
    def lower_bound(self):
        return self.__lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        if type(lower_bound) in [int, float]:
            self.__lower_bound = lower_bound
        else:
            raise RuntimeError

    @lower_bound.getter
    def lower_bound(self):
        return self.__lower_bound

    @property
    def upper_bound(self):
        return self.__upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        if type(upper_bound) in [int, float]:
            self.__upper_bound = upper_bound
        else:
            raise RuntimeError

    @upper_bound.setter
    def upper_bound(self):
        return self.__upper_bound

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, mu):
        if type(mu) in [int, float]:
            self.__mu = mu
        else:
            raise RuntimeError

    @mu.getter
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, sigma):
        if type(sigma) in [int, float]:
            self.__sigma = sigma
        else:
            raise RuntimeError

    @sigma.getter
    def sigma(self):
        return self.__sigma

    @property
    def exp_lambda(self):
        return self.__exp_lambda

    @exp_lambda.setter
    def exp_lambda(self, exp_lambda):
        if type(exp_lambda) in [int, float]:
            self.__exp_lambda = exp_lambda
        else:
            raise RuntimeError

    @exp_lambda.getter
    def exp_lambda(self):
        return self.__exp_lambda


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
            feature_value = self.__random.expovariate(self.__exp_lambda)

        if feature_value is not None:
            return feature_value
        else:
            raise RuntimeError


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    feature1 = Features()
    feature1.random_distribution_function = 'gauss'
    feature1.sigma = 10
    feature1.mu = 100


    feature2 = Features()
    feature2.random_distribution_function = 'expovariate'
    feature2.exp_lambda = -1

    data_pattern = DataPatternEnsembler()
    data_pattern.add_feature('x1', feature1)
    data_pattern.add_feature('x2', feature2)

    print(data_pattern)

    values = []

    while True:
        values.append(feature2.get_feature_value())

        if len(values) > 100000:
            break


    plt.hist(values, bins=100)
    plt.show()
