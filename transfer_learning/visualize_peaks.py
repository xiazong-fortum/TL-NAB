import plotly.graph_objects as go
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import numpy as np


data_file = "/mnt/data/TL-NAB/transfer_learning/data/full_data_plant2.csv"
df = pd.read_csv(data_file, index_col = 'timestamp')
data = df.iloc[:,[1]].iloc[:,0]
peaks, _ = find_peaks(data.values)

if len(peaks) > 1:
    peak_distances = np.diff(peaks)
    period = np.mean(peak_distances)
else:
    period = None


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data,
    mode='lines',
    name='Data',
    line=dict(color='blue'),
    opacity=0.8
))

fig.add_trace(go.Scatter(
    x=[data.index[p] for p in peaks],
    y=[data.iloc[p] for p in peaks],
    mode='markers',
    marker=dict(color='red', size=8, symbol='x', opacity=0.8),
    name='Peaks'
))

fig.add_trace(go.Scatter(
    x=[data.index[p] for p in peaks[1:]],
    y=peak_distances,
    mode='lines+markers',
    name='Peak Distances',
    yaxis='y2',
    line=dict(color='orange', dash='dash', width=2),
    opacity=0.8
))

fig.update_layout(
    title="Peak Detection with Interactive Dual Y-Axis",
    xaxis=dict(title='Timestamp', tickangle=90),
    yaxis=dict(title='Value', side='left'),
    yaxis2=dict(title='Peak Distance', overlaying='y', side='right'),
    width=1200,
    height=600,
    template='plotly_white'
)

fig.show()
