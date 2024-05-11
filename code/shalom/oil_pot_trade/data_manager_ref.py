import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
from torch.utils.data import DataLoader, TensorDataset

import settings

# 데이터 분할 비율 설정
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

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
    data['volume'][:-1].ffill()

    macd = ta.macd(data["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    data[["macd", "macd_signal", "macd_diff"]] = macd
    rsi = ta.rsi(data["close"], length=14)
    data["rsi"] = rsi
    rsi_signal = ta.rsi(data["close"], length=14, signal_indicator="ema", signal_period=9)
    data["rsi_signal"] = rsi_signal
    
    return data


def load_data(code):
    # 시퀀스 생성
    sequence_length = 64  #필요에 따라 수정 가능
    sequences_train = []
    sequences_valid = []
    sequences_test = []

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
    # df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    # df = df.fillna(method='ffill').reset_index(drop=True)

    # 날짜 형식 변환 및 필터링
    #df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]

    # 결측값 처리 및 인덱스 재설정

    df = df.ffill().reset_index(drop=True)

    # # 'date' 컬럼을 문자열 형식으로 변환 (필요에 따라)
    # df['date'] = df['date'].dt.strftime('%Y%m%d')

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]
    #df = df.drop(columns=['data'])
    # 학습 데이터 분리
    df1 = df.drop(columns=['date'])

    #training_data = df[COLUMNS_TRAINING_DATA_V1]

    # 데이터 분할 인덱스 계산
    train_index = int(len(df1) * train_ratio)
    valid_index = train_index + int(len(df1) * valid_ratio)

    # train_data, valid_data, test_data 분할
    train_data = df1.iloc[:train_index]
    valid_data = df1.iloc[train_index:valid_index]
    test_data = df1.iloc[valid_index:]

    # for i in range(len(train_data) - sequence_length + 1):
    #     sequence = train_data.iloc[i:i+sequence_length]
    #     sequences_train.append(sequence.values)
        
    # for i in range(len(valid_data) - sequence_length + 1):
    #     sequence = valid_data.iloc[i:i+sequence_length]
    #     sequences_valid.append(sequence.values)

    # for i in range(len(test_data) - sequence_length + 1):
    #     sequence = test_data.iloc[i:i+sequence_length]
    #     sequences_test.append(sequence.values)

    # # 리스트를 numpy 배열로 변환
    # sequences_train = np.array(sequences_train)
    # sequences_valid = np.array(sequences_valid)
    # sequences_test = np.array(sequences_test)

    # # 시퀀스의 마지막 데이터 포인트의 종가를 타겟 데이터로 사용
    # target_train = train_data['close'].iloc[sequence_length - 1:].values
    # target_valid = valid_data['close'].iloc[sequence_length - 1:].values
    # target_test = test_data['close'].iloc[sequence_length - 1:].values

    # # 텐서 변환
    # x_train = torch.Tensor(sequences_train)
    # y_train = torch.Tensor(target_train).unsqueeze(dim=1)  # 학습 데이터의 타겟(레이블) 예시
    # x_valid = torch.Tensor(sequences_valid)
    # y_valid = torch.Tensor(target_valid).unsqueeze(dim=1)  # 검증 데이터의 타겟(레이블) 예시
    # x_test = torch.Tensor(sequences_test)
    # y_test = torch.Tensor(target_test).unsqueeze(dim=1)  # 테스트 데이터의 타겟(레이블) 예시

    # # TensorDataset 생성
    # train_dataset = TensorDataset(x_train, y_train)
    # valid_dataset = TensorDataset(x_valid, y_valid)
    # test_dataset = TensorDataset(x_test, y_test)

    # # DataLoader 생성
    # train_loader = DataLoader(train_dataset, batch_size=group, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=group)
    # test_loader = DataLoader(test_dataset, batch_size=group)

    # # train_data, valid_data, test_data 반환
    # return chart_data, train_loader, valid_loader, test_loader, train_data.shape[1]
    return chart_data, train_data, valid_data, test_data, train_data.shape[1]