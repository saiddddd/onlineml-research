import time
from tqdm import tqdm

import pandas as pd
import requests
import json

from src.streaming_data_source.data_sourcing import DataSourcingKafka

api_url = 'http://127.0.0.1:5000/model/validation/'
headers = {'content-type': 'application/json'}


class InferenceDataSender:

    def __init__(self, load_df_path: str, index_name, label_name):
        # self._df = pd.read_csv('../../playground/quick_study/dummy_toy/dummy_data.csv')
        self._df = pd.read_csv(load_df_path)

        self._distinct_time = None

        # create index column, which is going to use in plot performance visualization
        if index_name is None:
            self._df.reset_index(inplace=True)
        else:
            try:
                self._df.set_index(index_name, inplace=True)
            except:
                print("can not set {} as index, column not found! going to reset_index".format(index_name))
                self._df.reset_index(inplace=True)

        # set label column name as lowercase 'y'
        try:
            self._df.rename(columns={label_name:'y'}, inplace=True)
        except:
            print("can not replace {} as new label name \'y\', please check provided label name is correct!".format(label_name))

        print(self._df.head(5))



    def run(self, time_series_column):

        kafka_sender = DataSourcingKafka()

        select_index = 0

        if time_series_column is not None:
            try:
                self._distinct_time = sorted(self._df[time_series_column].unique())
            except:
                self._distinct_time = None
                pass

        while True:

            x_axis_item = None
            if self._distinct_time is None:
                sub_df = self._df.iloc[select_index:select_index+100]
                x_axis_item = select_index
                select_index += 100
            else:
                try:
                    sub_df = self._df[self._df[time_series_column] == self._distinct_time[select_index]]
                    x_axis_item = self._distinct_time[select_index]
                    select_index += 1
                except:
                    break

            if len(sub_df.index) > 0:
                df_to_json = sub_df.to_json()
                wrap_to_send = {
                    'x_axis_name': x_axis_item,
                    'label_name': 'y',
                    'Data': str(df_to_json)
                }

                # response = requests.post(api_url, data=str(df_to_json), headers=headers)
                response = requests.post(api_url, data=json.dumps(wrap_to_send), headers=headers)

                # send to kafka
                if kafka_sender is not None:
                    sub_df.pop('index')
                    df_to_send = sub_df.rename(columns={'y': 'Y'})

                    for index, row in tqdm(df_to_send.iterrows(), total=df_to_send.shape[1]):
                        kafka_sender.send_to_kafka(row, 'testTopic')


                print(df_to_json)
                print(response.json())
                time.sleep(1)
            else:
                break

if __name__ == '__main__':
    sender = InferenceDataSender("../../playground/quick_study/dummy_toy/dummy_data.csv", 'index', 'Y')

    # sender = InferenceDataSender("../../data/stock_index_predict/eda_TW50_top30_append_test_start_from_2018.csv", 'index', 'LABEL')
    sender.run(time_series_column='Date')




# df = df.head(5)
# print(df)
#
# df_to_json = df.to_json()
# wrap_to_send = {'Data': str(df_to_json)}
#
# # response = requests.post(api_url, data=str(df_to_json), headers=headers)
# response = requests.post(api_url, data=json.dumps(df_to_json), headers=headers)
#
# print(df_to_json)
# print(response.json())
