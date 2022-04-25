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

                proba_list, is_target_list = self.inference(df)

                return Response(json.dumps(proba_list), status=200, headers={'content-type': 'application/json'})

            except Exception as e:
                e.with_traceback()
                print("can not extract data, please check!")
                abort(404)

    def run(self):
        self.app.run()


    def load_model(self, path):
        with open(path, 'rb') as f:
            self.__model = pickle.load(f)

    def inference(self, data: pd.DataFrame, proba_cut_point=0.5) -> (list, list):
        """
        The implementation of hoeffding tree model inference.
        return two list,\n
        the first one is the list of prediction probability if this inference is target (float). e.g. 0.35 -> 35% of change is target\n
        the second one is the list of prediction true/false if this inference is target (int). e.g. 1 -> this is target\n
        :param data: input data frame
        :param proba_cut_point: float
        :return: (prediction_probability in list, prediction_is_target)
        """

        pred_target_proba = []
        pred_is_target = []

        for index, raw in data.iterrows():

            result = self.__model.predict_proba_one(raw)

            try:

                pred_target_proba.append(result.get(1))

                if proba_cut_point is not None:
                    if result.get(1) >= proba_cut_point:
                        pred_is_target.append(1)
                    else:
                        pred_is_target.append(0)

            except Exception as e:
                pred_target_proba.append(None)
                pred_is_target.append(None)
                e.with_traceback()

        return pred_target_proba, pred_is_target







if __name__ == '__main__':

    online_model_serving = OnlineMachineLearningModelServing()
    online_model_serving.run()






