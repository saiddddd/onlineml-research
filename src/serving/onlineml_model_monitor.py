import os, glob
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import base64

from serving.onlineml_model_serving import OnlineMachineLearningModelServing

# Static method to show figure from local file


def tree_structure_inspect_display(display_fig_path: str):
    if len(display_fig_path) > 0:
        try:
            image_filename = display_fig_path  # image path
            encoded_image = base64.b64encode(open(image_filename, 'rb').read())
            return html.Div([
                html.H3("current Hoeffding Tree Structure"),
                html.Br(),
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    title='live check model structure'
                )
            ])
        except:
            return html.Div([
                html.H3("Model Structure inspection figure path is not correct: {}, fail to load image!".format(image_filename)),
            ])
    else:
        return html.Div([
            html.H3("invalid input, please support correct input figure path: grid_layout_tree_structure_inspect_display(<path>)"),
        ])


def grid_layout_tree_structure_inspect_display(display_fig_dir: str):

    def card_content_image_synthesis(image_filename: str):

        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        tree_name = image_filename.split('/')[-1].split('.')[0].split('_')[-1]
        card_content = [
            dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_image.decode()), top=True, style={'height':'200px', 'width':'100%'}),
            dbc.CardBody(
                [
                    html.H4("Tree Number: {}".format(tree_name), className="card-title"),
                    html.P(
                        "Some quick example text to build on the card title and "
                        "make up the bulk of the card's content.",
                        className="card-text",
                    ),
                    dbc.Button("Go somewhere", color="primary"),
                ]
            ),

        ]
        return card_content

    def build_img_card(image_filename):

        card = dbc.Card(
            card_content_image_synthesis(image_filename=image_filename),
            # style={"width": "100%", "height": "100%"},
            color='primary',
            outline=True
        )
        return card


    if os.path.isdir(display_fig_dir):
        listing_image = glob.glob(display_fig_dir+'*.png')
        listing_image.sort()

    return html.Div(
        [
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[1]), color='primary', outline=True)),
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[2]), color='primary', outline=True)),
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[3]), color='primary', outline=True))
                ],
                align='center',
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[4]), color='primary', outline=True)),
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[5]), color='primary', outline=True)),
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[6]), color='primary', outline=True))
                ],
                align='center',
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[7]), color='primary', outline=True)),
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[8]), color='primary', outline=True)),
                    dbc.Col(dbc.Card(card_content_image_synthesis(image_filename=listing_image[9]), color='primary', outline=True))
                ],
                align='center',
            ),

        ]
    )


class ModelPerformanceMonitor:

    _instance = None

    @staticmethod
    def get_instance():
        if ModelPerformanceMonitor._instance is None:
            ModelPerformanceMonitor._instance = ModelPerformanceMonitor()
        return ModelPerformanceMonitor._instance

    def __init__(self):

        self._online_ml_server = OnlineMachineLearningModelServing.get_instance()
        self.dash_display = Dash(__name__+'dash', external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                return tree_structure_inspect_display(
                    display_fig_path='../../output_plot/online_monitoring/current_tree_structure.png')

            elif pathname == '/test_grid_display':
                return grid_layout_tree_structure_inspect_display(
                    display_fig_dir='../../output_plot/online_monitoring/'
                )
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
