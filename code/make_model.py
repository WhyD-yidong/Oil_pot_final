import os
import csv
import time
import pyupbit
import datetime
import operator
import logging
import sys

max_data_cnt = 729
min_data_cnt = 240
#max_data_cnt = 364
offset_data_cnt = 120

os.environ['POT_BASE'] = 'D:\\oilpot'
from shalom.oil_pot_trade import settings
from shalom.oil_pot_trade import utils
from shalom.oil_pot_trade import data_manager
from shalom.oil_pot_trade.learners import A2CLearner

count = 0
single = 0
max_count = 0
coin_list=[]
except_list = []

today = datetime.datetime.today()
str_today = today.strftime("%Y%m%d")
start_day = today-datetime.timedelta(days=max_data_cnt)
str_start_day = start_day.strftime("%Y%m%d")


def count_rows(ticker):
    with open(os.path.join(settings.BASE_DIR,'data',f'{ticker}.csv'), 'r') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count


def get_next_ticker():
    cnt_idx = 0
    tickers = pyupbit.get_tickers(fiat='KRW')
    for idx in tickers:
        #print(cnt_idx,idx)
        cnt_idx += 1
        coin_list.append(idx)

    return cnt_idx-1


def update_date(ticker):
    now = datetime.datetime.now()
    if now.hour >= 9:
        today = datetime.datetime.today()
    else:
        today = datetime.datetime.today()-datetime.timedelta(days=1)
    
    str_t_day = today.strftime("%Y%m%d")
    data_count = count_rows(ticker)

    if data_count > max_data_cnt + offset_data_cnt:
        start_day = today-datetime.timedelta(days=max_data_cnt)
    else:
        start_day = today-datetime.timedelta(days=data_count-offset_data_cnt)


    str_s_day = start_day.strftime("%Y%m%d")
    return str_s_day, str_t_day


def check_except(code):
    if code in except_list:
        return 1
    else: return 0


max_count = get_next_ticker()

single = int(input("Enter 1 making single model"))
count = int(input("Enter coin_idx for starting: "))#int(sys.argv[1])

os.environ['POT_BACKEND'] = 'pytorch'
# 출력 경로 생성
output_path = os.path.join(settings.BASE_DIR, 'output')
if not os.path.isdir(output_path):
    os.makedirs(output_path)

# 최소/최대 단일 매매 금액 설정
min_trading_price = 100000
max_trading_price = 100000

# 공통 파라미터 설정
common_params = {'rl_method': 'a2c', 
    'net': 'lstm', 'num_steps': 32, 'lr': 0.0003,
    'balance': 100000, 'num_epoches': 45000,
    'discount_factor': 0.99, 'start_epsilon': 1,
    'output_path': output_path, 'reuse_models': False}


while True:
    now = datetime.datetime.now()
    coin_name = coin_list[count]
    data_count = count_rows(coin_name)
    
    if data_count < min_data_cnt or check_except(coin_name):
        print(count,coin_name,data_count,"Not enough min Data count or check except")
        #count += 1
        #if count >= max_count: count = 0
        break#continue
    
    if 1:#(now.hour == 00 or now.hour == 6 or now.hour == 12 or now.hour == 18) and now.minute == 00:
        str_start_day, str_today = update_date(coin_name)
        print(count, coin_name, count_rows(coin_name) - offset_data_cnt, str_start_day, str_today)
        chart_data, training_data = data_manager.load_data(
            coin_name, str_start_day, str_today)
        
        common_params.update({'stock_code': coin_name,
        'chart_data': chart_data, 
        'training_data': training_data,
        'min_trading_price': min_trading_price, 
        'max_trading_price': max_trading_price})
        
        value_network_name = coin_name+f'_value.mdl'
        policy_network_name = coin_name+f'_policy.mdl'

        value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
        policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)
        record_log_path = os.path.join(settings.BASE_DIR, 'logs')

        # 로그 기록 설정
        log_path = os.path.join(record_log_path, f'{coin_name}.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(format='%(message)s')
        logger = logging.getLogger(settings.LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        #stream_handler = logging.StreamHandler(sys.stdout)
        #stream_handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        #logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        #logger.info(params)

        learner = A2CLearner(**{**common_params, 
        'value_network_path': value_network_path, 
        'policy_network_path': policy_network_path})

        learner.run(learning='train')
        learner.save_models()

        count += 1
        if count >= max_count: count = 0
        if single != 0: break
    else:
        print("Sleep_now...  Ready for",count,coin_name,"at",now.hour,":",now.minute,":",now.second)
        if count >= max_count: count = 0
        time.sleep(1)