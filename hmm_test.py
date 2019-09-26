import numpy as np
import tushare as ts
from hmm_model import HmmModel
import pandas as pd
import codecs


stock_code = '000001.SZ'

start_date_train = '20150101'
end_date_train = '20181231'

# 导入金融数据
with codecs.open('token.txt','rb','utf-8') as f:
    token = f.read()
ts.set_token(token)
pro = ts.pro_api()
quotes = ts.pro_bar(ts_code=stock_code, start_date=start_date_train, end_date=end_date_train, adj='qfq')  # 前复权
#直接保存
quotes.to_csv('datas/'+stock_code+'_quotes_train.csv')
# 读csv
df1 = pd.read_csv('datas/'+stock_code+'_quotes_train.csv')  # martinindex   hdf文件 h5  共享内存
X1 = np.array([q for q in reversed(df1['pct_chg'])])  # 涨跌幅

flows = pro.moneyflow(ts_code=stock_code, start_date=start_date_train, end_date=end_date_train)
#直接保存
flows.to_csv('datas/'+stock_code+'_flows_train.csv')
# 读csv
df2 = pd.read_csv('datas/'+stock_code+'_flows_train.csv')
buy_lg_amount = np.array([q for q in reversed(df2['buy_lg_amount'])])      # 大单买入金额（万元）
sell_lg_amount = np.array([q for q in reversed(df2['sell_lg_amount'])])    # 大单卖出金额（万元）
buy_elg_amount = np.array([q for q in reversed(df2['buy_elg_amount'])])    # 特大单买入金额（万元）
sell_elg_amount = np.array([q for q in reversed(df2['sell_elg_amount'])])  # 特大单卖出金额（万元）
net_mf_amount = np.array([q for q in reversed(df2['net_mf_amount'])])      # 净流入额（万元）
total_amount = buy_lg_amount + buy_elg_amount + sell_lg_amount + sell_elg_amount  # 日总流动资金，计算方式为 (大单买入金额 + 大单卖出金额 + 特大单买入金额 + 特大单卖出金额)
X2 = net_mf_amount / total_amount   # 资金日净流入占当日所有流动资金的比例，这里计算方式为  净流入额 / 日总流动资金

X3 = []
for i in range(len(total_amount) - 1):
    X3.append((total_amount[i + 1] - total_amount[i]) / total_amount[i])
X3 = np.array(X3)     # 日总流动资金环比，这里计算方式为  (第二天日总流动资金 - 第一天日总流动资金) / 第一天日总流动资金

# 构建模型
model = HmmModel(state_num=9, sample_len=5)
# 训练模型，用X2和X3训练
model.train_model(X1[1:], X2[1:], X3)

# --------------------------------------------------------------------------------------------------------
# 测试模型准确率
# --------------------------------------------------------------------------------------------------------
start_date_test = '20190101'
end_date_test = '20191231'

quotes_test = ts.pro_bar(ts_code=stock_code, start_date=start_date_test, end_date=end_date_test, adj='qfq')
#直接保存
quotes_test.to_csv('datas/'+stock_code+'_quotes_test.csv')
# 读csv
df1_test = pd.read_csv('datas/'+stock_code+'_quotes_test.csv')
close_test = np.array([q for q in reversed(df1_test['close'])])  # 收盘价
dates_test = np.array([i for i in range(len(df1_test['close']))])  # 第多少天

high_test = np.array([q for q in reversed(df1_test['high'])])  # 最高价
low_test = np.array([i for i in reversed(df1_test['low'])])  # 最低价
mean_test = (high_test + low_test) / 2.0  # 均价，在使用  均价 = 成交额 / 成交量  计算时，数据和前一日的收盘价差别过大，似乎数据有问题，暂时用最高值和最低值的平均数代替

X1_test = np.array([q for q in reversed(df1_test['pct_chg'])])  # 涨跌幅

flows = pro.moneyflow(ts_code=stock_code, start_date=start_date_test, end_date=end_date_test)
#直接保存
flows.to_csv('datas/'+stock_code+'_flows_test.csv')
# 读csv
df2_test = pd.read_csv('datas/'+stock_code+'_flows_test.csv')
buy_lg_amount = np.array([q for q in reversed(df2_test['buy_lg_amount'])])
sell_lg_amount = np.array([q for q in reversed(df2_test['sell_lg_amount'])])
buy_elg_amount = np.array([q for q in reversed(df2_test['buy_elg_amount'])])
sell_elg_amount = np.array([q for q in reversed(df2_test['sell_elg_amount'])])
net_mf_amount = np.array([q for q in reversed(df2_test['net_mf_amount'])])
total_amount = buy_lg_amount + buy_elg_amount + sell_lg_amount + sell_elg_amount
X2_test = net_mf_amount / total_amount  # 资金日净流入占当日所有流动资金的比例，这里计算方式为  净流入额 / 日总流动资金

X3_test = []
for i in range(len(total_amount) - 1):
    X3_test.append((total_amount[i + 1] - total_amount[i]) / total_amount[i])
X3_test = np.array(X3_test)  # 日总流动资金环比，这里计算方式为  (第二天日总流动资金 - 第一天日总流动资金) / 第一天日总流动资金

# 回测模型，用X2和X3回测
model.back_test(X1_test[1:], close_test[1:], mean_test[1:], X2_test[1:], X3_test)

