import os
import traceback
import time
import json
import pickle
import requests
from datetime import datetime

import pandas as pd

from kafka import KafkaConsumer

from concurrent import futures

from river import tree
from river import ensemble

from tools.tree_structure_inspector import HoeffdingEnsembleTreeInspector
from tools.model import RiverClassifier, RiverRegressor, Model


class OnlineMachineLearningServer:

    def __init__(self):
        self.__server_status = None
        self.__model_store_dir = None
        self.__model_persist_file = None
        self.__model_persisting_process_status = None
        self.__kafka_consumer = None
        self.__model = None
        self.__trained_event_counter = 0

        self._tree_inspector = HoeffdingEnsembleTreeInspector(self.__model)

        self._init_kafka_consumer(
            connection_try_times=3
        )

    @property
    def server_status(self):
        return self.__server_status

    @server_status.getter
    def server_status(self):
        return self.__server_status

    @server_status.setter
    def server_status(self, status):
        if status == 'running' or status == 'stopped':
            self.__server_status = status
        else:
            self.__server_status = 'unknown'
            print("Status {} is not design in the server, set status as unknown".format(status))

    @property
    def model_store_dir(self):
        return self.__model_store_dir

    @model_store_dir.setter
    def model_store_dir(self, store: str):
        self.__model_store_dir = store

    @model_store_dir.getter
    def model_store_dir(self):
        return self.__model_store_dir

    @property
    def model_persist_file(self):
        return self.__model_persist_file

    @model_persist_file.setter
    def model_persist_file(self, file_name: str):
        self.__model_persist_file = file_name

    @model_persist_file.getter
    def model_persist_file(self):
        return self.__model_persist_file

    @property
    def model_persisting_process_status(self):
        return self.__model_persisting_process_status

    @model_persisting_process_status.setter
    def model_persisting_process_status(self, status):
        if status == 'flushing' or status == 'idle':
            self.__model_persisting_process_status = status
        else:
            self.__model_persisting_process_status = 'unknown'
            print("Status {} is not design in the server, set statis as unknown".format(status))



    def _init_kafka_consumer(self, connection_try_times=3):
        """
        initializing of kafka consumer
        :return:
        """
        self.__kafka_consumer = KafkaConsumer(
            'testTopic',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def load_model(self, load_model_path: str):
        """
        from provided dir to load exist model
        :param load_model_path:
        :return:
        """

        if os.path.isfile(load_model_path):
            print("checking {} is exist file".format(load_model_path))
            try:
                with open(load_model_path, 'rb') as f:
                    self.__model = pickle.load(f)
                    print('load exist model {} successfully.'.format(load_model_path))

                    """ model structure inspect, saving figure. """

            except FileNotFoundError:
                print("File not found! please check dir {} and pickle exist".format(load_model_path))


    def init_model(self, model: Model):
        """
        initializing of model
        :return:
        """
        self.__model = model

        ''' load model from pickle if pre-train model exist '''
        ''' model structure inspect, saving figure. '''


    def ensemble_model(self, ensemble_model_params):

        self.__model = self.__model.ensemble_model(model_encapsulate=ensemble.AdaBoostClassifier,
                                                   model_hyper_params=ensemble_model_params)


    def _save_model(self, save_file_path='', save_file_name=''):

        # check file name is exist, if not. provided a unique name for it
        if len(save_file_name) == 0:
            timestamp = time.time()
            save_file_name = 'persist_model_{}.pickle'.format(timestamp)
        # check folder exist, if not, stop this function
        if save_file_path[-1] != '/':
            save_file_path += '/'
        if os.path.isdir(save_file_path):
            model_persist_file = save_file_path+save_file_name
        else:
            raise FileNotFoundError('Can not found Model Persist Folder')
        with open(model_persist_file, 'wb') as out_file:
            try:
                pickle.dump(self.__model, out_file)
                print("Persist Model successfully at {}".format(
                    datetime.now().strftime('\r%Y-%m-%d %H:%M:%S')
                ))
            except Exception as e:
                e.with_traceback()
                print("Model Persisting Error, can not save model into file {}. Please check!".format(model_persist_file))

    def tree_inspect_and_dump(self, *args, **kwargs):

        self._tree_inspector.update_model(self.__model)
        self._tree_inspector.draw_tree(**kwargs)


    def train_model_by_one_polling_batch(self, polling_batch_data, label_name):
        """
        Responsible for extracting batch in one polling,
        iterating row by row.
        To do model updating while 1 polling batch is processed
        :param polling_batch_data: 1 block of polling
        :param label_name: label name to pop
        :return:
        """
        for record in polling_batch_data:
            '''Row level data extraction
            '''
            receive_data = record.value
            df = pd.DataFrame.from_dict(receive_data)

            print(df.dtypes)

            y = df.pop(label_name)

            print("going to fit by Model internally defined iteration and learn by one row")
            self.__model.fit(df, y)

        # Model has been updated
        self.__model_persisting_process_status = 'flushing'
        print('labeling model persisting status on')


    def stop(self):
        self.server_status = 'stopped'


    def run(self, consumer_run_mode='', label_name='LABEL'):

        print("running mode: {}; {}".format(consumer_run_mode, label_name))

        self.server_status = 'running'

        # To be configable in the future
        # MODE == iterating OR polling
        CONSUMER_RUN_MODE = consumer_run_mode
        LABEL_NAME = label_name

        if CONSUMER_RUN_MODE == 'polling':
            '''following block using polling method to do data extraction
            '''
            print("going to consumer kafka consumer in polling mode")
            try:
                while self.server_status == 'running':
                    data_polling_result = self.__kafka_consumer.poll(
                        timeout_ms=1000,
                        max_records=None,
                        update_offsets=True
                    )
                    for key, value in data_polling_result.items():

                        self.train_model_by_one_polling_batch(value, LABEL_NAME)

                    time.sleep(1)
            except:
                print("Unexpect Error Happen while running model training")
                self.stop()
                traceback.print_exc()

        elif CONSUMER_RUN_MODE == 'iteration':
            ''' following block using iteration method to run new event from consumer
            '''
            print("going to consumer kafka consumer in interation mode")
            for msg in self.__kafka_consumer:
                receive_data = msg.value

                row = pd.Series(receive_data)
                y = row.pop(LABEL_NAME)

                if row is None or y is None:
                    continue

                start_time = time.time()
                try:
                    self.__model.learn_one(row, y)
                except AttributeError:
                    print("attributeError from model training")
                    # print(traceback.format_exc())
                    pass
                end_time = time.time()

                self.__trained_event_counter += 1
                print(
                    '\r #{} Events Trained, learn_one time spend:{} milliseconds. Offset{}'.format(
                        self.__trained_event_counter, (end_time - start_time) * 1000, msg.offset),
                    end='',
                    flush=True
                )

                if self.__trained_event_counter % 500 == 0:
                    self.__model_persisting_process_status = 'flushing'
                    print('labeling model persisting status on')

        else:
            print("Cannot recognize running mode! please check!. Acceptance: 1. polling ; 2. iteration")
            raise RuntimeError

    def run_model_persist(self):

        def send_signal_load_model(url=''):
            url = url

            if len(url) > 1:
                response = requests.post(
                    url,
                    data='{"model_path":"../../model_store/testing_hoeffding_tree.pickle"}',
                    headers={'content-type': 'application/json'}
                )

        while self.server_status == 'running':

            time.sleep(1)
            print("going to persist model. status:{}".format(self.__model_persisting_process_status))


            if self.__model is not None and self.__model_persisting_process_status == 'flushing':
                try:
                    self._save_model(
                        save_file_path=self.model_store_dir,
                        save_file_name=self.model_persist_file
                    )

                    tree_inspect_params = {
                        'tree_index': None,
                        'output_fig_dir': '../../output_plot/web_checker_online_display/online_tree_inspection/',
                        'fig_file_name': 'current_tree_structure'
                    }
                    self.tree_inspect_and_dump(**tree_inspect_params)

                    timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
                    tree_inspect_params = {
                        'tree_index': 0,
                        'output_fig_dir': '../../output_plot/web_checker_historical_check/tree_inspection/',
                        'fig_file_name': '{}_tree_structure'.format(timestamp)
                    }
                    self.tree_inspect_and_dump(**tree_inspect_params)

                    time.sleep(3)
                    try:
                        send_signal_load_model("http://127.0.0.1:5000/model/")
                        print("\nSend load model request to model server")
                    except:
                        print("Can not send signal to serving part for load model api")
                    self.__model_persisting_process_status = 'idle'
                except FileNotFoundError:
                    print("Folder to persist model not found QQ! {}".format(os.getcwd()))
                except Exception:
                    print("An Unexpected Error happen in model persist thread!")
                    traceback.print_exc()

            elif self.__model_persisting_process_status == 'idle':
                print('Model is not been updated, persist process in idle status')
                time.sleep(10)



class OnlineMachineTrainerRunner:

    def __init__(self):

        print("Initialization of Online Machine Learning Service.")
        self.__server = OnlineMachineLearningServer()
        self.__server.model_store_dir = "../../model_store"
        self.__server.model_persist_file = "testing_hoeffding_tree.pickle"
        print("Online Machine Learning Service created.")

        self._pool = futures.ThreadPoolExecutor(2)
        self._future = None

    def start_online_ml_server(self):
        print("start online machine learning server")
        self._future = self._pool.submit(self.__server.run, consumer_run_mode='polling', label_name='LABEL')

    def start_persist_model(self):
        self._future = self._pool.submit(self.__server.run_model_persist)

    def stop_online_ml_server(self):
        print("stop online machine learning server")
        self.__server.stop()

    def load_model(self, load_model_path: str):
        self.__server.load_model(load_model_path=load_model_path)

    def init_model(self, model):
        self.__server.init_model(model)

    def ensemble_model(self, params):
        self.__server.ensemble_model(params)



if __name__ == "__main__":
    runner = OnlineMachineTrainerRunner()
    # runner.load_model(load_model_path='../../model_store/pretrain_model_persist/testing_hoeffding_tree_pretrain_model.pickle')

    # HoeffdingTree Classifier setting
    model_hyper_params = {
        'max_depth': 5,
        'split_criterion': 'gini',
        'split_confidence': 1e-2,
        'grace_period': 10,
        'seed': 0
    }

    ensemble_model_params = {
        'n_models': 100
    }
    runner.init_model(RiverClassifier(model_class=tree.HoeffdingAdaptiveTreeClassifier, model_hyper_params=model_hyper_params))
    runner.ensemble_model(ensemble_model_params)


    # # HoeffdingTree Regressor setting
    # model_hyper_params = {
    #     'grace_period': 50,
    #     'leaf_prediction': 'adaptive',
    #     'model_selector_decay': 0.3,
    #     'seed': 0
    # }
    # runner.init_model(RiverRegressor(model_class=tree.HoeffdingAdaptiveTreeRegressor, model_hyper_params=model_hyper_params))
    #

    runner.start_online_ml_server()
    runner.start_persist_model()
    # time.sleep(1000)
    # runner.stop_online_ml_server()
