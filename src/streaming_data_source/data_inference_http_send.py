import pandas as pd
import requests
import json

api_url = 'http://127.0.0.1:5000/model/validation/'
headers = {'content-type': 'application/json'}

df = pd.read_csv('../../playground/quick_study/dummy_toy/dummy_data.csv')

# df.pop('Y')

df = df.head(50)

df_to_json = df.to_json()
wrap_to_send = {'Data': str(df_to_json)}

# response = requests.post(api_url, data=str(df_to_json), headers=headers)
response = requests.post(api_url, data=json.dumps(df_to_json), headers=headers)

print(df_to_json)
print(response.json())
