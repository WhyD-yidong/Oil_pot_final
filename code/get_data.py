import os
import csv
import time
import pyupbit
import datetime
import pandas as pd

count = 0

os.environ['POT_BASE'] = 'C:\\'
from shalom.oil_pot_trade import settings
#from shalom.oil_pot_trade import utils
from shalom.oil_pot_trade import data_manager

print("data gethering start")

tickers = pyupbit.get_tickers(fiat='KRW')
for idx in tickers:
    count += 1
    print(count,idx)

    df = pyupbit.get_ohlcv(idx, interval="minute60", count=2400)
    time.sleep(0.5)
    df.to_csv(os.path.join(settings.BASE_DIR,'data',f'{idx}.csv'), index=True, header=False)

