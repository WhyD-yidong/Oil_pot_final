import os
import pandas as pd
import numpy as np
import pandas_ta as ta

from shalom.oil_pot_trade import settings


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume', 'value']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
    'macd','macd_signal','macd_diff','rsi','rsi_signal',
]

def preprocess(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:] - data['volume'][:-1])
        / data['volume'][:-1].ffill().bfill()
    )
    macd = ta.macd(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    data[["macd", "macd_signal", "macd_diff"]] = macd
    rsi = ta.rsi(data["close"], length=14)
    data["rsi"] = rsi
    rsi_signal = ta.rsi(data["close"], length=14, signal_indicator="ema", signal_period=9)
    data["rsi_signal"] = rsi_signal
    return data


def load_data(code, date_from, date_to):

    header = None
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', f'{code}.csv'),
        thousands=',', header=header, converters={'date': lambda x: str(x)})

    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'value']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = preprocess(df)
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    #df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.tail(2048)
    df = df.ffill().reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[COLUMNS_TRAINING_DATA_V1]
    
    return chart_data, training_data