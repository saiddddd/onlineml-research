import time
from datetime import datetime

from abc import ABC, abstractmethod
import pandas as pd
import random

import requests
import json

from kafka import KafkaProducer


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

class Generator(ABC):

    def __init__(self):

        self._gauss_random_generator = Features()
        self._gauss_random_generator.random_distribution_function = 'gauss'
        self._gauss_random_generator.sigma = 10
        self._gauss_random_generator.mu = 100

        self._exp_random_generator = Features()
        self._exp_random_generator.random_distribution_function = 'expovariate'
        self._exp_random_generator.exp_lambda = -1

        self._kafka_producer = None

    def init_kafka_producer(self, bootstrap_servers: str):
        self._kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers
        )
        print("Successfully initialized kafka producer")

    def generate_data(self, condition=0):
        x1 = self._gauss_random_generator.get_feature_value()
        x2 = self._gauss_random_generator.get_feature_value()
        x3 = self._gauss_random_generator.get_feature_value()
        x4 = self._gauss_random_generator.get_feature_value()
        x5 = self._gauss_random_generator.get_feature_value()
        x6 = self._gauss_random_generator.get_feature_value()

        exp_decay = self._exp_random_generator.get_feature_value()


        def get_y(condition):

            y = 0
            if condition == 0:
                if (x1 > 100 and x2 < 100) or (x1 < 100 and x3 < 100):
                    y = 1
                else:
                    y = 0
            else:
                if (x4 > 100 and x5 < 100) or (x4 < 100 and x6 < 100):
                    y = 1
                else:
                    y = 0

            if exp_decay > -3:
                """ to make some rare exception case noice, if expovariate(-1) function < -2. flip y
                """
                if y == 0:
                    y = 1
                else:
                    y = 0

            return y

        y = get_y(condition=condition)

        data_dict = {
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "X4": x4,
            "X5": x5,
            "X6": x6,
            "Y": y
        }

        row = pd.Series(data_dict)

        return row

    def send_to_kafka(self, topic: str, data):

        data_to_send = None
        if isinstance(data, str):
            data_to_send = bytes(data, 'utf-8')
        elif isinstance(data, pd.Series):
            data_to_send = bytes(str(data.to_json()), 'utf-8')
        elif isinstance(data, bytearray):
            data_to_send = data

        if data_to_send is not None:
            try:
                self._kafka_producer.send(topic, data_to_send)
                self._kafka_producer.flush()
            except Exception as e:
                e.with_traceback()

    def run_dataset_pump_to_kafka(self, iteration=1000, time_interval=0):

        for i in range(iteration):
            self.send_to_kafka('testTopic', self.generate_data(condition=0))

            if time_interval != 0:
                time.sleep(time_interval)


    def run_dataset_pump_to_inference_api(self, api_url: str):

        headers = {'content-type': 'application/json'}
        now = datetime.now()

        row = self.generate_data(condition=0)
        df_to_send = pd.DataFrame()
        df_to_send = df_to_send.append(row, ignore_index=True)
        # breakpoint()

        for i in range(999):
            row = self.generate_data(condition=0)
            # df_temp = pd.DataFrame(row)
            df_to_send = df_to_send.append(row, ignore_index=True)


        print(df_to_send.head(5))

        df_to_json = df_to_send.to_json()
        wrap_to_send = {
            'x_axis_name': now.strftime("%H:%M:%S"),
            'label_name': 'Y',
            'Data': str(df_to_json)
        }

        response = requests.post(api_url, data=json.dumps(wrap_to_send), headers=headers)
        print(response)

    # def generate_one(self):


if __name__ == '__main__':

    generator = Generator()
    generator.init_kafka_producer('localhost:9092')

    generator.run_dataset_pump_to_kafka()
    time.sleep(20)

    while True:
        generator.run_dataset_pump_to_kafka()
        time.sleep(5)
        generator.run_dataset_pump_to_inference_api('http://127.0.0.1:5000/model/validation/')

    #
    # from matplotlib import pyplot as plt
    #
    # gauss_random_generator = Features()
    # gauss_random_generator.random_distribution_function = 'gauss'
    # gauss_random_generator.sigma = 10
    # gauss_random_generator.mu = 100
    #
    # exp_random_generator = Features()
    # exp_random_generator.random_distribution_function = 'expovariate'
    # exp_random_generator.exp_lambda = -1
    #
    # # data_pattern = DataPatternEnsembler()
    # # data_pattern.add_feature('x1', gauss_random_generator)
    # # data_pattern.add_feature('x2', exp_random_generator)
    # # print(data_pattern)
    #
    # values = []
    #
    # while True:
    #     x1 = gauss_random_generator.get_feature_value()
    #     x2 = gauss_random_generator.get_feature_value()
    #     x3 = gauss_random_generator.get_feature_value()
    #     x4 = gauss_random_generator.get_feature_value()
    #     x5 = gauss_random_generator.get_feature_value()
    #     x6 = gauss_random_generator.get_feature_value()
    #
    #     exp_decay = exp_random_generator.get_feature_value()
    #
    #     def get_y(condition=0):
    #
    #         y = 0
    #         if condition == 0:
    #             if (x1 > 100 and x2 < 100) or (x1 < 100 and x3 < 100):
    #                 y = 1
    #             else:
    #                 y = 0
    #         else:
    #             if (x4 > 100 and x5 < 100) or (x4 < 100 and x6 < 100):
    #                 y = 1
    #             else:
    #                 y = 0
    #
    #         if exp_decay > -2:
    #             """ to make some rare exception case noice, if expovariate(-1) function < -2. flip y
    #             """
    #             if y == 0:
    #                 y = 1
    #             else:
    #                 y = 0
    #
    #         return y
    #
    #     y = get_y(condition=0)
    #
    #     data_dict = {
    #         "X1": x1,
    #         "X2": x2,
    #         "X3": x3,
    #         "X4": x4,
    #         "X5": x5,
    #         "X6": x6,
    #         "Y": y
    #     }
    #     values.append(y)
    #
    #     row = pd.Series(data_dict)
    #
    #     # print(row)
    #     # time.sleep(1)
    #
    #     if len(values) > 100000:
    #         break
    #
    #
    # plt.hist(values, bins=100)
    # plt.show()

