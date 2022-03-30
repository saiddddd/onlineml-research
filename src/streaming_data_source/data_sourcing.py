import abc
import time
import json

import pandas as pd

from kafka import KafkaProducer
from kafka.errors import KafkaError
import msgpack

from tools.data_loader import GeneralDataLoader

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

    def get_full_op_df(self):
        return self.__full_op_df


    def send_data_into_kafka(self, kafka_broker_location: str, send_topic: str, data):

        producer = KafkaProducer(
            bootstrap_servers=[kafka_broker_location]
            # value_serializer=lambda x: json.dumps(x).encode('utf-8')
            # value_serializer=msgpack.dumps
        )


        #TODO send data to Kafka
        print(data)
        producer.send(send_topic, value=data, key='training')


if __name__ == '__main__':

    data_generator = LocalDataGenerator('../../playground/quick_study/dummy_toy/dummy_data.csv')

    df = data_generator.get_full_op_df()

    for index, raw in df.iterrows():
        print(index, raw)
        data_encoded = raw.to_json().encode()
        data_generator.send_data_into_kafka('localhost:9092', 'testTopic', data_encoded)
        time.sleep(1)
        breakpoint()

