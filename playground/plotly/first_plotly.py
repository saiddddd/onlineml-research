from dash import Dash, html, dcc

import pandas as pd
import plotly.express as px

app = Dash(__name__)

#create dummy data to demo
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(
    df,
    x="Fruit",
    y="Amount",
    color="City",
    barmode="group"
)

app.layout = html.Div(
    children=[
        html.H1(children='''Hello Dash'''),
        html.Div(
            children='''Dash: A web application framework for your data.'''
        ),

        # given id for mapping if we want to add `interactivity later`
        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)