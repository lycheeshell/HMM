import multiprocessing
import pandas as pd
import numpy as np
from hmm_model import HmmModel
import time
import gc


def stock_hmm(stock, quotes, divide_date):
    print("开始 :", stock, ", 时间 :", time.ctime())
    train_quotes = quotes[quotes['tradeDate'] < divide_date]
    test_quotes = quotes[quotes['tradeDate'] >= divide_date]

    # 构建模型
    model = HmmModel(state_num=9, sample_len=5, stock_name=stock)

    # 训练模型
    close_train = np.array(train_quotes['closePrice'])
    day_num = len(close_train)
    change_train = np.zeros(day_num-1)  # 涨跌幅
    for i in range(day_num-1):
        change_train[i] = (close_train[i+1] - close_train[i]) / close_train[i] * 100
    turnover_vol_train = np.array(train_quotes['turnoverVol'])
    turnover_value_train = np.array(train_quotes['turnoverValue']) / 10000
    negmarket_value_train = np.array(train_quotes['negMarketValue']) / 1000000

    model.train_model(change_train, change_train, turnover_vol_train[1:], turnover_value_train[1:], negmarket_value_train[1:])

    # 回测模型
    close_test = np.array(test_quotes['closePrice'])
    day_num = len(close_test)
    change_test = np.zeros(day_num - 1)  # 涨跌幅
    for i in range(day_num - 1):
        change_test[i] = (close_test[i + 1] - close_test[i]) / close_test[i] * 100
    turnover_vol_test = np.array(test_quotes['turnoverVol'])
    turnover_value_test = np.array(test_quotes['turnoverValue'])
    mean_test = turnover_value_test / turnover_vol_test
    turnover_value_test = turnover_value_test / 10000
    negmarket_value_test = np.array(test_quotes['negMarketValue']) / 1000000

    model.back_test(change_test, close_test[1:], mean_test[1:], change_test, turnover_vol_test[1:], turnover_value_test[1:], negmarket_value_test[1:])

    del model
    gc.collect()
    print("结束 :", stock, ", =====================================时间 :", time.ctime())


if __name__ == '__main__':

    # secID,tradeDate,openPrice,closePrice,lowestPrice,highestPrice,accumAdjFactor(累积前复权因子),turnoverVol(成交量),turnoverValue(成交金额),negMarketValue(流通市值)
    quotes_dataframe = pd.read_hdf("datas/quotes.h5", "quotes")

    all_quotes = quotes_dataframe['secID'].unique()

    divide_date = '2019-01-01'  # 该时间之前的数据为训练集，之后的数据为测试集

    print("main 开始时间 :", time.ctime())

    # cpu_num = multiprocessing.cpu_count()
    # print("cpu num : ", cpu_num)

    pool = multiprocessing.Pool(processes=2)

    for stock in all_quotes[0:6]:
        quotes = quotes_dataframe[quotes_dataframe['secID'] == stock]
        pool.apply_async(stock_hmm, (stock, quotes, divide_date))

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
