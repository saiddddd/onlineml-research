import os
import traceback
import time
import json
import pickle
import requests
from concurrent import futures
from datetime import datetime

import pandas as pd

from kafka import KafkaConsumer

from river import ensemble
from river import tree

from concurrent import futures

from tools.tree_structure_inspector import HoeffdingEnsembleTreeInspector

class OnlineMachineLearningServer:

    def __init__(self):
        self.__server_status = None
        self.__model_persisting_process_status = None
        self.__kafka_consumer = None
        self.__model = None
        self.__trained_event_counter = 0

        self._init_kafka_consumer(
            connection_try_times=3
        )
        self._init_model()

    @property
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

    def _init_model(self):
        """
        initializing of model
        :return:
        """

        self.__model = ensemble.AdaBoostClassifier(
            model=(
                tree.HoeffdingAdaptiveTreeClassifier(
                    max_depth=3,
                    split_criterion='gini',
                    split_confidence=1e-2,
                    grace_period=10,
                    seed=0
                )
            ),
            n_models=10,
            seed=42
        )

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


    def train_model_by_one_row(self, receive_data: pd.Series, label_name: str):
        """
        Passing row level data from polling block to do model training by line
        :param receive_data: passing row level data (1 row)
        :param label_name: label name to pop
        :return:
        """
        row = pd.Series(receive_data)
        y = row.pop(label_name)

        '''model training => learn one
        '''
        start_time = time.time()
        try:
            self.__model.learn_one(row, y)
        except Exception as e:
            print(traceback.format_exc())
        end_time = time.time()

        self.__trained_event_counter += 1

        print(
            '\r #{} Events Trained, learn_one time spend:{} milliseconds'.format(
                self.__trained_event_counter, (end_time - start_time) * 1000),
            end='',
            flush=True
        )

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
            self.train_model_by_one_row(receive_data, label_name=label_name)

        # Model has been updated
        self.__model_persisting_process_status = 'flushing'


    def stop(self):
        self.__server_status = 'stopped'


    def run(self, consumer_run_mode='', label_name=''):

        print("running mode: {}; {}".format(consumer_run_mode, label_name))

        self.__server_status = 'running'

        # To be configable in the future
        # MODE == iterating OR polling
        CONSUMER_RUN_MODE = consumer_run_mode
        LABEL_NAME = label_name

        if CONSUMER_RUN_MODE == 'polling':
            '''following block using polling method to do data extraction
            '''
            print("going to consumer kafka consumer in polling mode")
            while self.__server_status == 'running':
                data_polling_result = self.__kafka_consumer.poll(
                    timeout_ms=1000,
                    max_records=None,
                    update_offsets=True
                )
                for key, value in data_polling_result.items():
                    self.train_model_by_one_polling_batch(value, LABEL_NAME)
                time.sleep(1)

        elif CONSUMER_RUN_MODE == 'iteration':
            ''' following block using iteration method to run new event from consumer
            '''
            print("going to consumer kafka consumer in interation mode")
            for msg in self.__kafka_consumer:
                receive_data = msg.value

                row = pd.Series(receive_data)
                y = row.pop(LABEL_NAME)

                start_time = time.time()
                try:
                    self.__model.learn_one(row, y)
                except Exception as e:
                    print(traceback.format_exc())
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

        while self.__server_status == 'running':

            time.sleep(10)
            print("going to persist model. status:{}".format(self.__model_persisting_process_status))


            if self.__model is not None and self.__model_persisting_process_status == 'flushing':
                try:
                    self._save_model(
                        save_file_path='../../model_store',
                        save_file_name='testing_hoeffding_tree.pickle'
                    )
                    tree_inspector = HoeffdingEnsembleTreeInspector(self.__model)
                    # tree_inspector.draw_tree(0, '../../output_plot/tree_inspect/')


                    # updating current tree structure to /output_plot/online_monitoring
                    # for checker inspection
                    # tree_inspector.draw_tree(0, '../../output_plot/online_monitoring/', 'current_tree_structure')
                    tree_inspector.draw_tree(None, '../../output_plot/online_monitoring/tree_inspection/', 'current_tree_structure')
                    time.sleep(3)
                    try:
                        send_signal_load_model('http://127.0.0.1:5000/model/')
                    except:
                        print("Can not send signal to serving part for load model api")
                    self.__model_persisting_process_status = 'idle'
                except FileNotFoundError:
                    print("Folder to persist model not found QQ! {}".format(os.getcwd()))

            elif self.__model_persisting_process_status == 'idle':
                print('Model is not been updated, persist process in idle status')
                time.sleep(10)



class OnlineMachineTrainerRunner:

    def __init__(self):

        print("Initialization of Online Machine Learning Service.")
        self.__server = OnlineMachineLearningServer()
        print("Online Machine Learning Service created.")

        self._pool = futures.ThreadPoolExecutor(2)
        self._future = None

    def start_online_ml_server(self):
        print("start online machine learning server")
        self._future = self._pool.submit(self.__server.run, consumer_run_mode='polling', label_name='Y')

    def start_persist_model(self):
        self._future = self._pool.submit(self.__server.run_model_persist)

    def stop_online_ml_server(self):
        print("stop online machine learning server")
        self.__server.stop()




if __name__ == "__main__":
    runner = OnlineMachineTrainerRunner()
    runner.start_online_ml_server()
    runner.start_persist_model()
    # time.sleep(1000)
    # runner.stop_online_ml_server()
