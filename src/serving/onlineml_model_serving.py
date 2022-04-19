from flask import Flask, request, Response, abort
import pickle
import pandas as pd
from pandas import json_normalize
import json

class OnlineMachineLearningModelServing:



    def __init__(self):
        self.app = Flask(__name__)

        self.__model = None

        @self.app.route('/model/', methods=['POST'])
        def model_load():

            try:
                raw_request_message = request.get_json()
                read_path = raw_request_message['model_path']
                print(read_path)
                self.load_model(read_path)

                result_json = {'message': 'Success'}

                return Response(result_json, status=200, mimetype="application/json")

            except:
                print("get model path error, please check key is correct!")
                abort(404)
                pass

        @self.app.route('/model/inference/', methods=['POST'])
        def model_inference():

            try:

                receive_data_payload = request.get_json()
                df = pd.read_json(receive_data_payload)

                self.inference(df)

                return Response(status=200)

            except Exception as e:
                e.with_traceback()
                print("can not extract data, please check!")
                abort(404)

    def run(self):
        self.app.run()


    def load_model(self, path):
        with open(path, 'rb') as f:
            self.__model = pickle.load(f)

    def inference(self, data):

        for index, raw in data.iterrows():
            # print(raw)
            result = self.__model.predict_proba_one(raw)
            print(result)




if __name__ == '__main__':

    online_model_serving = OnlineMachineLearningModelServing()
    online_model_serving.run()






