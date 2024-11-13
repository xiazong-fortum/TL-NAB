import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# 读取数据
data_file = "shf_all_2324.csv"
dataframe = pd.read_csv(data_file)


# Initialize Dash app
app = dash.Dash(__name__)

# Layout with multi-select dropdown for choosing multiple columns
app.layout = html.Div([
    html.H1("时间序列数据展示"),
    dcc.Dropdown(
        id="column-selector",
        options=[{"label": col, "value": col} for col in dataframe.columns[1:]],  # Exclude the time column
        value=[dataframe.columns[1]] if len(dataframe.columns) > 1 else [],  # Default selection
        multi=True,
        clearable=False
    ),
    dcc.Graph(id="time-series-graph"),
])

# Callback function to update graph with selected columns
@app.callback(
    Output("time-series-graph", "figure"),
    [Input("column-selector", "value")]
)
def update_graph(selected_columns):
    # Ensure selected_columns is not empty
    if not selected_columns:
        return {"data": [], "layout": go.Layout(title="请在下拉菜单中选择至少一列")}

    traces = []
    try:
        # Generate a trace for each selected column
        for column in selected_columns:
            if column in dataframe.columns:
                y_values = pd.to_numeric(dataframe[column], errors='coerce')  # Convert to numeric, handle non-numeric gracefully
                traces.append(go.Scatter(x=dataframe['time_window_start'], y=y_values, mode='lines', name=column))

        layout = go.Layout(
            title="多列数据展示",
            xaxis=dict(title="时间"),
            yaxis=dict(title="数值"),
            legend=dict(title="选择的列")
        )

        return {"data": traces, "layout": layout}

    except Exception as e:
        print(f"Error in callback: {e}")
        return {"data": [], "layout": go.Layout(title="Error: Unable to generate graph")}

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)