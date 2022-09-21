import datetime
import traceback

from flask import Flask, request, Response, abort
import pickle
from concurrent import futures
import pandas as pd
import numpy as np
from pandas import json_normalize
import json

from sklearn.ensemble import RandomForestClassifier
# Model validation related
from sklearn.metrics import recall_score, accuracy_score, f1_score

from tools.tree_structure_inspector import HoeffdingEnsembleTreeInspector
from tools.model_perform_visualization import PredictionProbabilityDist


class OnlineMachineLearningModelServing:

    # design in Singleton Pattern
    _instance = None

    @staticmethod
    def get_instance():
        if OnlineMachineLearningModelServing._instance is None:
            OnlineMachineLearningModelServing._instance = OnlineMachineLearningModelServing()
        return OnlineMachineLearningModelServing._instance

    # internal constructor
    def __init__(self):
        """
        Initialization of model Serving.
        """
        self._pool = futures.ThreadPoolExecutor(2)
        self._future = None

        self.app = Flask(__name__)

        self.__model = None
        self.__batch_model = RandomForestClassifier(
            n_estimators=10,
            criterion='gini',
            verbose=1
        )
        # df_batch_train = pd.read_csv("../../data/stock_index_predict/eda_TW50_top30_append_2010_2017.csv")
        # df_batch_train.drop(['Date', 'DailyReturn'], axis=1, inplace=True)
        # df_batch_train = pd.read_csv("../../playground/quick_study/dummy_toy/dummy_data_training.csv")
        # y = df_batch_train.pop('LABEL')
        # self.__batch_model.fit(df_batch_train, y)

        # variable for metrics display
        self.__x_axis = []
        self.__appending_acc = []
        self.__appending_recall = []
        self.__appending_f1 = []
        self.__appending_mae = []

        # variable for metrics display
        self.__batch_appending_acc = []
        self.__batch_appending_f1 = []
        self.__batch_appending_mae = []

        # last prediction probability result and target answer
        self.__last_pred_proba = None
        self.__last_y_true = None

        @self.app.route('/model/', methods=['POST'])
        def load_model_api():
            """
            API to call server to load persist model from local file system
            Ingress https request
            :return:
            """
            try:
                raw_request_message = request.get_json()
                read_path = raw_request_message['model_path']
                print(read_path)
                self.load_model(read_path)
                print("Load model from {} successfully".format(read_path))

                result_json = {'message': 'Success'}

                return Response(result_json, status=200, mimetype="application/json")

            except pickle.UnpicklingError as e:
                # ts = traceback.extract_stack()
                print(e.with_traceback())
                print("Model Unpickling Error, please check the target is correct model pickle file.")
                abort(404)
                pass
            except FileNotFoundError:
                print("The file: \'{}\' Not Found!, please check.".format(read_path))
                abort(404)
                pass
            finally:
                print("Finish of handling this load model request")

        # @self.app.route('/model/inference/', methods=['POST'])
        # def model_inference_api() -> Response:
        #     """
        #     Model inference api, request contain dataset which want to do predict
        #     label is not expect to be access in this api.
        #     :return: http response with prediction result in payload in json format
        #     """
        #     try:
        #         x_axis_item, df = extract_http_data_payload(request)
        #         proba_list, is_target_list = self.inference(df)
        #         return Response(json.dumps(proba_list), status=200, headers={'content-type': 'application/json'})
        #
        #     except Exception as e:
        #         e.with_traceback()
        #         print("can not extract data, please check!")
        #         abort(404)


        @self.app.route('/model/validation/', methods=['POST'])
        def model_validation():

            try:
                x_axis_item, label_name, df = extract_http_data_payload(request)

                y = df.pop(label_name)

                # go model inference and get result
                # proba_list, is_target_list = self.inference(df)
                proba_list, is_target_list = self.__model.inference_proba(df)
                # Update last prediction proba and target y
                self.__last_pred_proba = proba_list
                self.__last_y_true = y

                # ========================================= #
                # Performing prediction probability quality #
                # ========================================= #
                # # drawing prediction probability distribution and saving figure
                pred_proba_nparray = np.array(proba_list)
                time_stamp = datetime.datetime.now().strftime('%HH-%MM-%SS')

                model_pred_dist_drawer = PredictionProbabilityDist(
                    pred_proba_nparray,
                    y
                )
                draw_fig = model_pred_dist_drawer.draw_proba_dist_by_true_false_class_seperated()
                draw_fig.savefig('../../output_plot/web_checker_historical_check/model_pred_proba_distribution/pred_proba_check_{}.png'.format(time_stamp))
                draw_fig.savefig('../../output_plot/web_checker_online_display/online_pred_proba_distribution/pred_proba_check.png')

                acc = accuracy_score(y, is_target_list)
                recall = recall_score(y, is_target_list)
                f1 = f1_score(y, is_target_list)

                self.__x_axis.append(x_axis_item)

                self.__appending_acc.append(acc)
                self.__appending_recall.append(recall)
                self.__appending_f1.append(f1)

                print("Accuracy: {}\n recall-rate: {}\n f1 score: {}\n".format(acc, recall, f1))

                try:
                    df.drop(["Date", "DailyReturn"], axis=1, inplace=True)
                except:
                    pass
                batch_model_predict = self.__batch_model.predict(df)
                batch_acc = accuracy_score(y, batch_model_predict)
                batch_f1 = f1_score(y, batch_model_predict)
                self.__batch_appending_acc.append(batch_acc)
                self.__batch_appending_f1.append(batch_f1)

                print("batch model prediction Accuracy: {}\n f1 score: {}\n".format(batch_acc, batch_f1))

                return Response(
                    json.dumps(
                        {"accuracy": acc, "recall-rate": recall, "f1 score": f1}
                    ),
                    status=200,
                    headers={'content-type': 'application/json'}
                )

            except Exception as e:
                e.with_traceback()
                print("can not extract data, please check!")
                abort(404)


        def extract_http_data_payload(request_from_http: request):
            """
            extract dataframe from http request
            :param request_from_http: http requests
            :return: dataframe
            """

            receive_data_payload = request_from_http.get_json()

            x_axis_item = receive_data_payload['x_axis_name']
            label_name = receive_data_payload['label_name']
            df = pd.read_json(receive_data_payload['Data'])

            return x_axis_item, label_name, df

    def get_model_tree(self):

        model_inspector = HoeffdingEnsembleTreeInspector(self.__model)
        model_tree = model_inspector.get_tree_g(0)
        model_tree.render()

    def get_x_axis(self):
        return self.__x_axis

    def get_accuracy(self):
        return self.__appending_acc

    def get_f1_score(self):
        return self.__appending_f1

    def get_mae(self):
        return self.__appending_mae

    def get_batch_acc(self):
        return self.__batch_appending_acc

    def get_batch_f1(self):
        return self.__batch_appending_f1

    def get_batch_mae(self):
        return self.__batch_appending_mae

    def get_recall_rate(self):
        return self.__appending_recall

    def get_last_pred_proba(self):
        return self.__last_pred_proba

    def get_last_y_true(self):
        return self.__last_y_true

    def run(self):
        self._future = self._pool.submit(self.app.run)

    def load_model(self, path: str):
        """
        The implementation of the trigger acceptance api to load model from specify file path.
        :param path: path of model to load
        :return:
        """
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

