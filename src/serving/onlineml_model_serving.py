import datetime
import traceback

from flask import Flask, request, Response, abort
import pickle
from concurrent import futures
import pandas as pd
import numpy as np
from pandas import json_normalize
import json

# Model validation related
from sklearn.metrics import recall_score, accuracy_score, f1_score
# Metrics render via web page
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from tools.tree_structure_inspector import HoeffdingEnsembleTreeInspector

from matplotlib import pyplot as plt


def draw_analyze_proba_distribution(pred_proba: np.array, is_target_list: pd.Series, fig_save_path: str):

    pred_proba_result_true_class = pred_proba[is_target_list == 1]
    pred_proba_result_false_class = pred_proba[is_target_list == 0]

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle('{}pred_proba_distribution'.format(''))
    plt.subplot(131)
    plt.hist(pred_proba_result_true_class, bins=50, alpha=0.5, label='Y True')
    plt.hist(pred_proba_result_false_class, bins=50, alpha=0.5, label='Y False')
    plt.yscale('log')
    plt.title('stacking prediction proba in both class')
    plt.xlabel('pred proba')
    plt.ylabel('statistics')
    plt.grid()
    plt.legend()
    plt.subplot(132)
    plt.hist(pred_proba_result_true_class, bins=50)
    plt.yscale('log')
    plt.title('Y True class prediction proba. dist.')
    plt.xlabel('pred proba')
    plt.ylabel('statistics')
    plt.grid()
    plt.subplot(133)
    plt.hist(pred_proba_result_false_class, bins=50)
    plt.yscale('log')
    plt.title('Y False class prediction proba. dist.')
    plt.xlabel('pred proba')
    plt.ylabel('statistics')
    plt.grid()
    # fig.savefig(fig_save_path)
    # plt.savefig(fig_save_path)
    return fig


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

        # variable for metrics display
        self.__x_axis = []
        self.__appending_acc = []
        self.__appending_recall = []
        self.__appending_f1 = []

        # last prediction probability result and target answer
        self.__last_pred_proba = None
        self.__last_y_true = None

        self.dash_display = Dash(__name__+'dash')

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

        @self.app.route('/model/inference/', methods=['POST'])
        def model_inference_api() -> Response:
            """
            Model inference api, request contain dataset which want to do predict
            label is not expect to be access in this api.
            :return: http response with prediction result in payload in json format
            """
            try:
                x_axis_item, df = extract_http_data_payload(request)
                proba_list, is_target_list = self.inference(df)
                return Response(json.dumps(proba_list), status=200, headers={'content-type': 'application/json'})

            except Exception as e:
                e.with_traceback()
                print("can not extract data, please check!")
                abort(404)


        @self.app.route('/model/validation/', methods=['POST'])
        def model_validation():

            try:
                x_axis_item, label_name, df = extract_http_data_payload(request)

                y = df.pop(label_name)

                # go model inference and get result
                proba_list, is_target_list = self.inference(df)
                # Update last prediction proba and target y
                self.__last_pred_proba = proba_list
                self.__last_y_true = y

                pred_proba_nparray = np.array(proba_list)
                time_stamp = datetime.datetime.now().strftime('%HH-%MM-%SS')
                draw_fig = draw_analyze_proba_distribution(pred_proba_nparray, y, '../../output_plot/web_checker_historical_check/model_pred_proba_distribution/pred_proba_check_{}.png'.format(time_stamp))
                #saving figure
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

        # Multiple components can update everytime interval gets fired.
        @self.dash_display.callback(Output('live-update-graph', 'figure'),
                                    Input('interval-component', 'n_intervals'))
        def update_graph_live(n):
            x_list = self.get_x_axis()
            y_list = self.get_accuracy()
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=x_list, y=y_list, name='Accuracy',
                line=dict(color='firebrick', width=4)
            ))
            fig_acc.update_layout(
                title='Accuracy Trend Plot',
                xaxis_title='Iteration(s)',
                yaxis_title='Accuracy'
            )
            return fig_acc

        def extract_http_data_payload(request_from_http: request) -> pd.DataFrame:
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

    def get_recall_rate(self):
        return self.__appending_recall

    def get_last_pred_proba(self):
        return self.__last_pred_proba

    def get_last_y_true(self):
        return self.__last_y_true

    def run(self):
        self._future = self._pool.submit(self.app.run)
        # self.app.run()

    def run_dash(self):

        colors = {
            'background': '#111111',
            'text': '#7FDBFF'
        }

        self.dash_display.layout = html.Div(
            style={'backgroundColor': colors['background']},
            children=[
                html.H1(
                    children='Online Machine Learning Checker',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                    }
                ),
                html.Div(
                    children=
                    '''
                    Model Performance live-updating monitor
                    ''',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                    }
                ),
                dcc.Graph(id='live-update-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0
                )
            ])

        # self.dash_display.run_server()
        self._future = self._pool.submit(self.dash_display.run_server)


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



class ModelPerformanceMonitor:

    # Design in Singleton Pattern
    _instance = None

    @staticmethod
    def get_instance():

        if ModelPerformanceMonitor._instance is None:
            ModelPerformanceMonitor._instance = ModelPerformanceMonitor()
        return ModelPerformanceMonitor._instance

    def __init__(self):

        self._model_ml = OnlineMachineLearningModelServing.get_instance()
        self.dash_display = Dash(__name__ + 'dash')

        # Multiple components can update everytime interval gets fired.
        @self.dash_display.callback(Output('live-update-graph', 'figure'),
                                    Input('interval-component', 'n_intervals'))
        def update_graph_live(n):
            x_list = self._model_ml.get_x_axis()
            y_list = self._model_ml.get_accuracy()
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=x_list, y=y_list, name='Accuracy',
                line=dict(color='firebrick', width=4)
            ))
            fig_acc.update_layout(
                title='Accuracy Trend Plot',
                xaxis_title='Iteration(s)',
                yaxis_title='Accuracy'
            )
            return fig_acc

    def run_dash(self):

        colors = {
            'background': '#111111',
            'text': '#7FDBFF'
        }

        self.dash_display.layout = html.Div(
            style={'backgroundColor': colors['background']},
            children=[
                html.H1(
                    children='Online Machine Learning Checker',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                    }
                ),
                html.Div(
                    children=
                    '''
                    Model Performance live-updating monitor
                    ''',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                    }
                ),
                dcc.Graph(id='live-update-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0
                )
            ])

        self.dash_display.run_server()
        # self._future = self._pool.submit(self.dash_display.run_server)




if __name__ == '__main__':

    online_model_serving = OnlineMachineLearningModelServing.get_instance()
    online_model_serving.run()
    model_checker = ModelPerformanceMonitor.get_instance()
    model_checker.run_dash()
    # online_model_serving.run_dash()






