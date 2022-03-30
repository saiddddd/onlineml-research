import time
import json
from abc import ABC
import pandas as pd
from kafka import KafkaConsumer

from river import tree
from river import ensemble

class DataAcquisitor(ABC):
    """
    Abstraction of Data ACQ, concrete implement by cases
    """

    def __init__(self):
        self.__data_acquisitor_status = None

    @property
    def acquisitor_status(self):
        return self.__data_acquisitor_status

    @acquisitor_status.setter
    def acquisitor_status(self, running_status):
        self.__data_acquisitor_status = running_status

    def get_data_acquisitor_status(self):

        return self.acquisitor_status()


class KafkaDataAcquisitor(DataAcquisitor):

    def __init__(self, bootstrap_server: str, topic: str):

        super(KafkaDataAcquisitor).__init__()

        self.__bootstrap_server = bootstrap_server
        self.__topic = topic

        self.__kafka_consumer = KafkaConsumer(
            self.__topic,
            bootstrap_servers=[self.__bootstrap_server],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

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

    def run(self):

        self.acquisitor_status = 'running'
        print('Kafka Data ACQ is running')

        # while self.acquisitor_status == 'running':

            # data = self.__kafka_consumer.poll(timeout_ms=0, max_records=1, update_offsets=True)
            # print(data)
            # time.sleep(1)

        for msg in self.__kafka_consumer:

            recive_data = msg.value
            row = pd.read_json(json.dumps(recive_data), typ='series', orient='records')
            y = row.pop('Y')

            self.__model.learn_one(row, y)
            print('Model learn one')

            if self.acquisitor_status == 'stopped':
                break

        print("Kafka Data Acquisitor is stopped")


    def stop(self):

        print("calling stop function")
        self.acquisitor_status = 'stopped'
