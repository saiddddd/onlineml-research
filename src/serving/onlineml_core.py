import time
from concurrent import futures

from kafka import KafkaConsumer
from src.streaming_data_source.data_aquisitor import KafkaDataAcquisitor


class OnlineMachineLearningTrainer:

    def __init__(self):
        self.__kafka_data_acq = KafkaDataAcquisitor(
            bootstrap_server='localhost:9092',
            topic='testTopic'
        )

        self._pool = futures.ThreadPoolExecutor(3)
        self._future = self._pool.submit(self.__kafka_data_acq.run)

    def stop_runner(self):
        self.__kafka_data_acq.stop()
        self._future.cancel()

        print('going to join thread')
        self._pool.shutdown()


if __name__ == "__main__":
    runner = OnlineMachineLearningTrainer()
    time.sleep(1000)
    runner.stop_runner()