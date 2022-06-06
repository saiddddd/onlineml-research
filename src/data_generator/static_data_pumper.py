import pandas
import os
from tqdm import tqdm
import time
import requests
import json

import pandas as pd
from kafka import KafkaProducer


class DataPumper:

    def __init__(self):
        self._df = None
        self._kafka_producer = None

    def init_kafka_producer(self, bootstrap_servers: str):
        self._kafka_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers
        )
        print("Successfully initialized kafka producer")

    def load_data_from_csv(self, file_path: str):

        if os.path.isfile(file_path) and file_path.split('.')[-1] == 'csv':
            self._df = pd.read_csv(file_path)
            print("load data successfully")
        elif file_path.split('.')[-1] != 'csv':
            print("{} is not a csv file!".format(file_path))
        else:
            print("{} not found!".format(file_path))

    def show_df(self, n_row_to_show=5):
        if self._df is not None:
            print(self._df.head(n_row_to_show))
        else:
            print("Error, the dataframe is not created yet! please load data first!")

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


    def run_dataset_pump_to_kafka(self, start_row=0, end_row=None, time_interval=0):

        slicing_df = self._df.iloc[start_row: end_row]

        for index, row in tqdm(slicing_df.iterrows(), total=slicing_df.shape[1]):
            self.send_to_kafka('testTopic', row)

            if time_interval != 0:
                time.sleep(time_interval)

    def run_dataset_pump_to_inference_api(self, api_url: str, start_row=0, end_row=None, batch_size=10, is_online_learning=False):

        headers = {'content-type': 'application/json'}

        slicing_df = self._df.iloc[start_row: end_row]
        sending_offset = 0

        while True:
            sub_df_to_send = slicing_df[sending_offset:sending_offset+batch_size]
            if len(sub_df_to_send.index) > 0:
                df_to_json = sub_df_to_send.to_json()
                wrap_to_send = {
                    'x_axis_name': sending_offset,
                    'label_name': 'Y',
                    'Data': str(df_to_json)
                }

                response = requests.post(api_url, data=json.dumps(wrap_to_send), headers=headers)
                print(response)

                for index, row in tqdm(slicing_df[sending_offset:sending_offset+batch_size].iterrows(), total=batch_size):
                    self.send_to_kafka('testTopic', row)

                sending_offset += batch_size
                time.sleep(1)
            else:
                break




if __name__ == "__main__":

    pumper = DataPumper()
    pumper.init_kafka_producer(bootstrap_servers='localhost:9092')
    pumper.load_data_from_csv("../../playground/quick_study/dummy_toy/dummy_data.csv")
    pumper.show_df()
    pumper.run_dataset_pump_to_kafka(0, 600)
    time.sleep(30)
    pumper.run_dataset_pump_to_inference_api(
        'http://127.0.0.1:5000/model/validation/',
        3000,
        end_row=None,
        batch_size=100
    )
