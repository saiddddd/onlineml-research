import rpyc
from rpyc.utils.server import ThreadedServer
from kafka import KafkaConsumer

import json

class KafkaConsumerServer(rpyc.Service):

    def __init__(self):
        self.__kafka_consumer = KafkaConsumer(
            'testTopic',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )


    def exposed_get_consumer(self):

        return self.__kafka_consumer


if __name__ == "__main__":

    server = ThreadedServer(
        KafkaConsumerServer,
        port=12345,
        protocol_config={
            'allow_public_attrs': True
        }
    )

    server.start()