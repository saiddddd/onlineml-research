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
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
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

    def run_dataset_pump_to_inference_api(self, api_url: str, start_row=0, end_row=None, batch_size=10,
                                          label_name='Y', time_series_column_name=''):

        headers = {'content-type': 'application/json'}

        slicing_df = self._df.iloc[start_row: end_row]
        sending_offset = 0

        distinct_time = None
        if len(time_series_column_name) >= 1:
            try:
                distinct_time = sorted(slicing_df[time_series_column_name].str[:7].unique())
                print(distinct_time)
            except:
                print("Column {} not found!".format(time_series_column_name))


        # initialization of slicing data frame and offset
        # slicing_df = None
        # sending_offset = 0

        # if distinct_time is None:
        # print("time series is not provided, used row number based slicing")


        while True:


            if distinct_time is None:
                sub_df_to_send = slicing_df[sending_offset:sending_offset+batch_size]
            else:
                sub_df_to_send = slicing_df[slicing_df[time_series_column_name].str[:7] == distinct_time[sending_offset]]


            if len(sub_df_to_send.index) > 0:
                '''
                ================================================================
                Sending to Online ML server Prediction API to do model inference
                ================================================================
                '''
                x_axis_name = sending_offset
                if distinct_time is not None:
                    x_axis_name = distinct_time[sending_offset]
                    sub_df_to_send.drop(columns=[time_series_column_name], inplace=True)


                df_to_json = sub_df_to_send.to_json()
                wrap_to_send = {
                    'x_axis_name': x_axis_name,
                    'label_name': label_name,
                    'Data': str(df_to_json)
                }
                response = requests.post(api_url, data=json.dumps(wrap_to_send), headers=headers)
                print(response)

                '''
                ===================================================
                 Sending back to kafka to do online model training
                ===================================================
                '''
                # for index, row in tqdm(sub_df_to_send.iterrows(), total=sub_df_to_send.shape[0]):
                #     self.send_to_kafka('testTopic', row)
                self.send_to_kafka('testTopic', sub_df_to_send)

                '''
                ==============================
                 sender offset move one step
                ==============================
                '''
                if distinct_time is None:
                    sending_offset += batch_size
                else:
                    sending_offset += 1
                time.sleep(10)
            else:
                break




if __name__ == "__main__":


    def run_dummy_data():

        print("Running Dummy Dataset")

        pumper = DataPumper()
        pumper.init_kafka_producer(bootstrap_servers='localhost:9092')
        pumper.load_data_from_csv("../../playground/quick_study/dummy_toy/dummy_data_testing.csv")

        # # training
        # pumper.run_dataset_pump_to_kafka(0, 5000)

        # time.sleep(100)

        # prediction
        pumper.run_dataset_pump_to_inference_api(
            'http://127.0.0.1:5000/model/validation/',
            9000,
            end_row=None,
            batch_size=200,
            label_name='LABEL',
        )


    def run_stock_data():

        print("Running Stock Dataset")
        pumper = DataPumper()
        pumper.init_kafka_producer(bootstrap_servers='localhost:9092')
        pumper.load_data_from_csv("../../data/stock_index_predict/eda_TW50_top30_append_2010_2017.csv")

        # # training
        # pumper.run_dataset_pump_to_kafka(0, 5000)

        # time.sleep(100)

        # prediction
        pumper.run_dataset_pump_to_inference_api(
            'http://127.0.0.1:5000/model/validation/',
            9000,
            end_row=None,
            batch_size=200,
            label_name='LABEL',
            time_series_column_name='Date'
        )


    # run_dummy_data()
    run_stock_data()

    # pumper = DataPumper()
    # pumper.init_kafka_producer(bootstrap_servers='localhost:9092')
    # # pumper.load_data_from_csv("../../playground/quick_study/dummy_toy/dummy_data_testing.csv")
    # pumper.load_data_from_csv("../../data/stock_index_predict/eda_TW50_top30_append_2010_2017.csv")
    # pumper.show_df()
    #
    #
    # # # training
    # # pumper.run_dataset_pump_to_kafka(0, 5000)
    #
    #
    # # time.sleep(100)
    #
    # # prediction
    # pumper.run_dataset_pump_to_inference_api(
    #     'http://127.0.0.1:5000/model/validation/',
    #     9000,
    #     end_row=None,
    #     batch_size=200,
    #     label_name='LABEL',
    #     time_series_column_name='Date'
    # )
