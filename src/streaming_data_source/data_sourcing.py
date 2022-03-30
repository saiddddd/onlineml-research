import abc
import time
import json

import pandas as pd

from kafka import KafkaProducer
from kafka.errors import KafkaError
import msgpack

from tools.data_loader import GeneralDataLoader


class DataSourcingKafka:

    def __init__(self):

        self.__kafka_producer = None
        self._init_kafka_producer()

    def _init_kafka_producer(self, *args):

        #TODO kafka producer configuration should let user setting in the future
        broker_host_name = 'localhost:9092'

        self.__kafka_producer = KafkaProducer(
            bootstrap_servers=broker_host_name
        )

    def send_to_kafka(self, data, target_topic):

        data_to_send = None
        if isinstance(data, str):
            print("input data is string")
            data_to_send = bytes(data, 'utf-8')
        elif isinstance(data, pd.Series):
            print("input data is pandas series (row)")
            data_to_send = bytes(str(data.to_json()), 'utf-8')
        elif isinstance(data, bytearray):
            print("input data is byte array")
            data_to_send = data

        if data_to_send is not None:
            try:
                #TODO target topic should be configurable in the future
                target_topic = 'testTopic'
                self.__kafka_producer.send(target_topic, data_to_send)
                self.__kafka_producer.flush()

            except Exception as e:
                e.with_traceback()


class BaseGenerator(abc.ABC):

    def __init__(self):
        """
        An abstraction of Streaming Data Generator,
        generator help online ml demonstration.
        Running static data as online streaming.
        Generator could be seperated into two parts.
        1. ingress data from Persisted Real-Data / Simulated Dummy Data / Real-Time Data from Web-API or else
        2. Pumping data into streaming Pipeline

        Data preprocessing could / could not be done in generator. depends on needed, preprocess can be done in the whale
        """

        raise NotImplementedError("implement data generator with concrete usage")

    def ingress_data(self):

        raise NotImplementedError("implement data ingress function by case")

    def flush_data(self):

        raise NotImplementedError("implement data flush function by case")


class LocalDataGenerator(BaseGenerator):

    def __init__(self, local_file_path: str):

        super(LocalDataGenerator).__init__()

        self.__general_data_loader = GeneralDataLoader(local_file_path)
        self.__full_op_df = self.__general_data_loader.get_full_df()
        self.__data_sourcer_to_kafka = DataSourcingKafka()


    def get_full_op_df(self):
        return self.__full_op_df


    def send_data_into_kafka(self):

        #TODO send data to Kafka
        for index, row in self.__full_op_df.iterrows():
            self.__data_sourcer_to_kafka.send_to_kafka(row, 'testTopic')
            # time.sleep(3)
            # producer.send(send_topic, value=data, key='training')


if __name__ == '__main__':

    data_generator = LocalDataGenerator('../../playground/quick_study/dummy_toy/dummy_data.csv')
    data_generator.send_data_into_kafka()

    # df = data_generator.get_full_op_df()
    #
    # for index, raw in df.iterrows():
    #     print(index, raw)
    #     data_encoded = raw.to_json().encode()
    #     data_generator.send_data_into_kafka('localhost:9092', 'testTopic', data_encoded)
    #     time.sleep(1)
    #     breakpoint()

