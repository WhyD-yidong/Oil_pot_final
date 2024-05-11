import os
import csv
import time
import pyupbit
import datetime
import operator
import pandas as pd
import math

os.environ['POT_BASE'] = 'C:\\'
os.environ['POT_BACKEND'] = 'pytorch'

from shalom.oil_pot_trade import settings
from shalom.oil_pot_trade import data_manager


access = "YNAJxn9de4KObEKNaUgIQjVLDRxqcnYLlvIUzxxf"
secret = "VNBnA3zQw3mxF2wPmFjAlQ5Tk1v67HmAJ1z1JoLl"

min_krw = 5050
Debug_mode = False
timestamp_trading = False

def get_ticker():
    tickers = pyupbit.get_tickers(fiat='KRW')
    return tickers

def get_balance(ticker): #without KRW
    # """잔고 조회"""
    basic_name = ticker
    if ticker != "KRW":
        basic_name = ticker[4:]
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == basic_name:
            if b['balance'] != None:
                return float(b['balance'])
    return 0.0

def get_buy_price(ticker):  #without KRW
    # """매수가격 조회"""
    basic_name = ticker[4:]
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == basic_name:
            if b['avg_buy_price'] != None:
                return float(b['avg_buy_price'])
    return 0.0

def get_cur_price(ticker):
    # """현재가격 조회"""
    return pyupbit.get_current_price(ticker)


def get_Holdcoin(ticker):
    balance = get_balance(ticker)
    price = get_buy_price(ticker)

    if balance * price > min_krw:
        return balance
    else: return 0.0

def sell_market_order(name, balance):
    if Debug_mode:
        print("Debug Sell",coin_name,"amount",balance)
    else:
        print(coin_name, ':::', HoldCoin, "\t===========> SELL OUT",slot_count,"/",all_tk)
        upbit.sell_market_order(name, balance)
    

def buy_market_order(name, price):
    if Debug_mode:
        print("Debug Buy",coin_name,"amount",price)
    else:
        print(coin_name, ':::', HoldCoin, "\t<=========== BUY IN",slot_count,"/",all_tk)
        upbit.buy_market_order(name, price * 0.9995)

def save_data(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute60", count=1)

    if Debug_mode:
        print("Debug save_data",ticker)
    else:
        df.to_csv(os.path.join(settings.BASE_DIR,'data',f'{ticker}.csv'), index=True, header=False, mode='a')
        time.sleep(0.5)

def model_exists(ticker):
    chk_file_name = os.path.join(settings.BASE_DIR,'models',ticker+f'_policy.mdl')
    if os.path.exists(chk_file_name):
        global value_network_path, policy_network_path
        value_network_name = coin_name+f'_value.mdl'
        policy_network_name = coin_name+f'_policy.mdl'

        value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
        policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)
        return True
    else: return False

def count_models():
    files=os.listdir(os.path.join(settings.BASE_DIR,'models'))
    return len(files)/2

def run_predict(ticker):
    from shalom.oil_pot_trade.learners import  ReinforcementLearner, A2CLearner

    # 최소/최대 단일 매매 금액 설정
    min_trading_price = 100000
    max_trading_price = 100000
    today = datetime.datetime.today()
    str_start_day = str_today = today.strftime("%Y%m%d")

    chart_data, training_data = data_manager.load_data(
        ticker, str_start_day, str_today)

    # 공통 파라미터 설정
    common_params = {'rl_method': 'a2c', 
        'net': 'lstm', 'num_steps': 32, 'lr': 0.0003,
        'balance': 100000, 'num_epoches': 45000, 
        'discount_factor': 0.99, 'start_epsilon': 1,
        'output_path': output_path, 'reuse_models': 'predict'}
    
    common_params.update({'stock_code': ticker,
        'chart_data': chart_data, 
        'training_data': training_data,
        'min_trading_price': min_trading_price, 
        'max_trading_price': max_trading_price})
    
    learner = A2CLearner(**{**common_params, 
        'value_network_path': value_network_path, 
        'policy_network_path': policy_network_path})
    
    result = learner.predict()
    tuple_result = result[-1]
    #print(tuple_result)
    return tuple_result[1]

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

# 출력 경로 생성
output_path = os.path.join(settings.BASE_DIR, 'output')
if not os.path.isdir(output_path):
    os.makedirs(output_path)

idx = get_ticker()
all_tk = len(idx)

while True:
    try:
        now = datetime.datetime.now()
        if now.minute == 00:
            available_amount = get_balance('KRW')
            usage_amount = upbit.get_amount('ALL')
            numofmodel = count_models()
            unit_amount = (available_amount + usage_amount) / numofmodel
            slot_count = 0
            print()
            print("BALANCE - - - ava:",available_amount,"used:",usage_amount,"unit_amount", unit_amount, "models", numofmodel)
            print()
            
            for coin_name in idx:
                save_data(coin_name)
                if model_exists(coin_name) == True:
                    BuynSell = run_predict(coin_name)  #0:BUY 1:SELL
                    HoldCoin = get_Holdcoin(coin_name)

                    if BuynSell == 0 and HoldCoin == 0.0:
                        buy_market_order(coin_name, unit_amount)
                        slot_count += 1
                    elif BuynSell == 1 and HoldCoin != 0.0:
                        sell_market_order(coin_name, HoldCoin)
                        slot_count -= 1
                        if slot_count < 0 : slot_count = 0
                    else:
                        print(coin_name, "No condition")
                else:
                    print(coin_name,"No model")


        else:
            print(f'\r --- Wait trading ---: {now.hour:2d}:{now.minute:2d}:{now.second:2d}', end='')
            time.sleep(1)

    except Exception as e:
        print(e)
        time.sleep(5)
            