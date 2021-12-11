import argparse

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", help="Path to pickle file")
parser.add_argument("--base_url", default="/~kawamura/public/my_ner_dash/", help="Base url")
parser.add_argument("--max_text_length", type=int, default=30, help="Base url")
parser.add_argument("--num_sample", type=int, default=10, help="Number of samples to show")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--host", default="127.0.0.1", help="Host")
parser.add_argument("--port", default=8060, help="Port")
parser.add_argument("--data_path", default='/mnt/elm/kawamura', help="Port")
args = parser.parse_args()

app = dash.Dash(__name__, url_base_pathname=args.base_url, external_stylesheets=[dbc.themes.BOOTSTRAP])
topic_trend_df = pd.read_pickle(f'{args.data_path}/topic_trend_df.pkl')
target_trend_df = pd.read_pickle(f'{args.data_path}/target_trend_df.pkl')
fig_topic = px.line(topic_trend_df, x=topic_trend_df.index, y=topic_trend_df.columns)
fig_target = px.line(target_trend_df, x=target_trend_df.index, y=target_trend_df.columns)

app.layout = html.Div(
    [
        dbc.Row(
            html.Div(
                [
                    html.H1("topicの頻度変化"),
                    dcc.Graph(figure=fig_topic)
                ]
            ),
        ),
        dbc.Row(
            html.Div(
                [
                    html.H1("targetの頻度変化"),
                    dcc.Graph(figure=fig_target)
                ]
            ),
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=args.debug, host=args.host, port=args.port)
