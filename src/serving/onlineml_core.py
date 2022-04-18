import os
import time
import json
import pickle
from concurrent import futures

import pandas as pd

from kafka import KafkaConsumer

from river import ensemble
from river import tree

from concurrent import futures

class OnlineMachineLearningServer:

    def __init__(self):
        self.__server_status = None
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
        if os.path.isfile(save_file_path):
            model_persist_file = save_file_path+save_file_name
        else:
            raise FileNotFoundError('Can not found Model Persist Folder')

        with open(model_persist_file, 'wb') as out_file:
            try:
                pickle.dump(self.__model, out_file)
            except Exception as e:
                e.with_traceback()
                print("Model Persisting Error, can not save model into file {}. Please check!".format(model_persist_file))


    def train_model_by_one_polling_batch(self, polling_batch_data):

        for record in polling_batch_data:
            receive_data = record.value

            row = pd.Series(receive_data)
            # row = pd.read_json(json.dumps(
            #     dict(receive_data),
            #     typ='series',
            #     orient='records'
            # ))
            y = row.pop('Y')

            start_time = time.time()
            self.__model.learn_one(row, y)
            end_time = time.time()

            self.__trained_event_counter += 1

            print(
                '\r #{} Events Trained, learn_one time spend:{} milliseconds'.format(
                    self.__trained_event_counter, (end_time - start_time) * 1000),
                end='',
                flush=True
            )

    def stop(self):
        self.__server_status = 'stopped'


    def run(self):

        self.__server_status = 'running'

        while self.__server_status == 'running':

            print("pooling message from kafka")

            data_polling_result = self.__kafka_consumer.poll(
                timeout_ms=1000,
                max_records=None,
                update_offsets=True
            )

            for key, value in data_polling_result.items():
                self.train_model_by_one_polling_batch(value)

            print("end of this pooling, sleep 1 second")
            time.sleep(1)


class OnlineMachineTrainerRunner:

    def __init__(self):

        print("Initialization of Online Machine Learning Service.")
        self.__server = OnlineMachineLearningServer()
        print("Online Machine Learning Service created.")

        self._pool = futures.ThreadPoolExecutor(2)
        self._future = None

    def start_online_ml_server(self):
        print("start online machine learning server")
        self._future = self._pool.submit(self.__server.run)

    def stop_online_ml_server(self):
        print("stop online machine learning server")
        self.__server.stop()




if __name__ == "__main__":
    runner = OnlineMachineTrainerRunner()
    runner.start_online_ml_server()
    # time.sleep(1000)
    # runner.stop_online_ml_server()
