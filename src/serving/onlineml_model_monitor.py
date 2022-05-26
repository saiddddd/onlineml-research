
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import base64

from serving.onlineml_model_serving import OnlineMachineLearningModelServing


def grid_layout_tree_structure_inspect_display():
    image_filename = '../../output_plot/online_monitoring/current_tree_structure.png'  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return html.Div([
        html.H3("current Hoeffding Tree Structure"),
        html.Br(),
        html.Img(
            src='data:image/png;base64,{}'.format(encoded_image.decode()),
            title='live check model structure'
        )
    ])

class ModelPerformanceMonitor:

    _instance = None

    @staticmethod
    def get_instance():
        if ModelPerformanceMonitor._instance is None:
            ModelPerformanceMonitor._instance = ModelPerformanceMonitor()
        return ModelPerformanceMonitor._instance

    def __init__(self):

        self._online_ml_server = OnlineMachineLearningModelServing.get_instance()
        self.dash_display = Dash(__name__+'dash')

        # Multiple components can update everytime interval gets fired.
        @self.dash_display.callback(Output('live-update-acc-graph', 'figure'),
                                    Input('interval-component', 'n_intervals'))
        def update_acc_trend_live(n):
            x_list = self._online_ml_server.get_x_axis()
            y_list = self._online_ml_server.get_accuracy()
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

        @self.dash_display.callback(Output('live-update-f1-graph', 'figure'),
                                    Input('interval-component', 'n_intervals'))
        def update_f1_trend_live(n):
            x_list = self._online_ml_server.get_x_axis()
            y_list = self._online_ml_server.get_f1_score()
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                x=x_list, y=y_list, name='f1-score',
                line=dict(color='blue', width=4)
            ))
            fig_f1.update_layout(
                title='f1 scores Trend Plot',
                xaxis_title='Iteration(s)',
                yaxis_title='Accuracy'
            )
            return fig_f1

        @self.dash_display.callback(Output('model-performance-page', 'children'),
                                    [Input('url', 'pathname')])
        def display_page(pathname):
            if pathname == '/inference_performance':
                return html.Div([
                    dcc.Graph(id='live-update-acc-graph'),
                    dcc.Graph(id='live-update-f1-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=1 * 1000,  # in milliseconds
                        n_intervals=0
                    ),
                ])
            elif pathname == '/current_model_structure':
                return grid_layout_tree_structure_inspect_display()


            else:
                return html.Div([
                    dcc.Link(children='Go to Model Inference Performance Page.',
                             href='/inference_performance'
                             ),
                    html.Br(),
                    dcc.Link(children='Go to Model Structure inspection page.',
                             href='/current_model_structure'
                             )
                ])


    def run_dash(self):

        colors = {
            'background': '#111111',
            'text': '#7FDBFF'
        }

        self.dash_display.layout = html.Div(

            [
                dcc.Location(id='url', refresh=False),

                # first division for title and web mete message
                html.Div(
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
                    ]
                ),

                # definition of plot at display function
                html.Div(id="model-performance-page")

            ]
        )

        self.dash_display.run_server()
        # self._future = self._pool.submit(self.dash_display.run_server)
