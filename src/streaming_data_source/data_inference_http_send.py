import time

import pandas as pd
import requests
import json

api_url = 'http://127.0.0.1:5000/model/validation/'
headers = {'content-type': 'application/json'}

df = pd.read_csv('../../playground/quick_study/dummy_toy/dummy_data.csv')

# df.pop('Y')

select_index = 0

while True:
    sub_df = df.iloc[select_index:select_index+100]
    print(len(sub_df.index))
    select_index += 100

    if len(sub_df.index) > 0:
        df_to_json = sub_df.to_json()
        wrap_to_send = {'Data': str(df_to_json)}

        # response = requests.post(api_url, data=str(df_to_json), headers=headers)
        response = requests.post(api_url, data=json.dumps(df_to_json), headers=headers)

        print(df_to_json)
        print(response.json())
        time.sleep(1)
    else:
        break



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
