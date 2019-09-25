import numpy as np
import tushare as ts
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

state_num = 10
sample_len = 5
stock_code = '000001.SZ'
start_date = '20110101'
end_date = '20181231'

# 导入金融数据
ts.set_token('94fb9d6bf6205a73f0337eb7d397be9911e06eaf902cb2074bc36e9b')
pro = ts.pro_api()
quotes = ts.pro_bar(ts_code=stock_code, start_date=start_date, end_date=end_date, adj='qfq')  # 前复权
X1 = np.array([q for q in reversed(quotes['pct_chg'])])  # 涨跌幅

flows = pro.moneyflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
buy_lg_amount = np.array([q for q in reversed(flows['buy_lg_amount'])])
sell_lg_amount = np.array([q for q in reversed(flows['sell_lg_amount'])])
buy_elg_amount = np.array([q for q in reversed(flows['buy_elg_amount'])])
sell_elg_amount = np.array([q for q in reversed(flows['sell_elg_amount'])])
net_mf_amount = np.array([q for q in reversed(flows['net_mf_amount'])])
total_amount = buy_lg_amount + buy_elg_amount + sell_lg_amount + sell_elg_amount
X2 = net_mf_amount / total_amount

X3 = []
for i in range(len(total_amount) - 1):
    X3.append((total_amount[i + 1] - total_amount[i]) / total_amount[i])
X3 = np.array(X3)

X = np.column_stack([scale(X1[1:]), scale(X2[1:]), scale(X3)])

days = X.shape[0]

lengths = [sample_len] * int(days / sample_len)
if (days % sample_len != 0):
    lengths.append(days % sample_len)
lengths = np.array(lengths)

model = GaussianHMM(n_components=state_num, n_iter=500, tol=0.001)
model.fit(X, lengths)  # 训练模型————学习问题
print("输出根据数据训练出来的π")
print(model.startprob_)
print("输出根据数据训练出来的A")
print(model.transmat_)
hidden_states = model.predict(X)  # 估计状态序列————解码问题
# print('hidden_states', hidden_states, 'len: ', len(hidden_states))

judge_result = np.zeros((state_num))
max_state = []
min_state = []
for i in range(state_num):
    for temp in range(len(hidden_states)):
        if i == hidden_states[temp]:
            judge_result[i] = judge_result[i] + X1[temp]
for i in range(len(judge_result)):
    if judge_result[i] > 30:
        max_state.append(i)
    elif judge_result[i] < -30:
        min_state.append(i)

print("求和结果: ", judge_result)
print("涨隐状态的索引:", max_state, ", 跌隐状态的索引: ", min_state)

next_state = np.zeros((state_num))
for i in range(state_num):
    next_state[i] = np.argmax(model.transmat_[i])
print('每个状态所对应的下一个状态: ', next_state)

# --------------------------------------------------------------------------------------------------------
# 测试模型准确率
# --------------------------------------------------------------------------------------------------------
start_date_test = '20190101'
end_date_test = '20190831'

quotes_test = ts.pro_bar(ts_code=stock_code, start_date=start_date_test, end_date=end_date_test, adj='qfq')

close_test = np.array([q for q in reversed(quotes_test['close'])])
dates_test = np.array([i for i in range(len(quotes_test['close']))])

# vol_test = np.array([q for q in reversed(quotes_test['vol'])])
# amount_test = np.array([i for i in reversed(quotes_test['amount'])])
high_test = np.array([q for q in reversed(quotes_test['high'])])
low_test = np.array([i for i in reversed(quotes_test['low'])])
mean_test = (high_test + low_test) / 2.0  # 在使用  均价= 成交额 / 成交量  计算时，数据和前一日的收盘价差别过大，似乎数据有问题，暂时用最高值和最低值的平均数代替

X1_test = np.array([q for q in reversed(quotes_test['pct_chg'])])  # 涨跌幅

flows = pro.moneyflow(ts_code=stock_code, start_date=start_date_test, end_date=end_date_test)
buy_lg_amount = np.array([q for q in reversed(flows['buy_lg_amount'])])
sell_lg_amount = np.array([q for q in reversed(flows['sell_lg_amount'])])
buy_elg_amount = np.array([q for q in reversed(flows['buy_elg_amount'])])
sell_elg_amount = np.array([q for q in reversed(flows['sell_elg_amount'])])
net_mf_amount = np.array([q for q in reversed(flows['net_mf_amount'])])
total_amount = buy_lg_amount + buy_elg_amount + sell_lg_amount + sell_elg_amount
X2_test = net_mf_amount / total_amount

X3_test = []
for i in range(len(total_amount) - 1):
    X3_test.append((total_amount[i + 1] - total_amount[i]) / total_amount[i])
X3_test = np.array(X3_test)

X_test = np.column_stack([scale(X1_test[1:]), scale(X2_test[1:]), scale(X3_test)])
days_test = X_test.shape[0]

print('days_test: ', days_test)
print('X1_test: ', len(X1_test))
print('close_test: ', len(close_test))
print('mean_test: ', len(mean_test))

correct_num = 0
base_money = close_test[sample_len - 1]
base_money_fee = base_money
base = 1
base_fee = base
model_line = [base_money]
model_line_fee = [base_money]
print('初始金额 : ', base_money)

buyed = 1
buy_num = 0
hold_num = 0
sell_num = 0
empty_num = 0
for i in range(sample_len, days_test):
    arr_test = X_test[i - sample_len:i, :]
    states_test = model.predict(arr_test)
    predict_state = next_state[states_test[sample_len - 1]]
    if (predict_state in max_state and X1_test[i+1] > 0) or (predict_state in min_state and X1_test[i+1] < 0) or (
            predict_state not in max_state and predict_state not in min_state and X1_test[i+1] < 1 and X1_test[i+1] > -1):
        correct_num += 1
    if (predict_state in max_state) and (not buyed): # 预测涨， 买入
        buyed = 1
        buy_num += 1
        rate_temp = (mean_test[i+1] - close_test[i+1-1]) / close_test[i+1-1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
        base = base * (1 + rate_temp)
        base_money = base_money * (1 + rate_temp)
        base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
        base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
    elif (predict_state in min_state) and buyed: # 预测跌，抛出
        buyed = 0
        sell_num += 1
        rate_temp = (mean_test[i+1] - close_test[i+1 - 1]) / close_test[i+1 - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
        base = base * (1 + rate_temp)
        base_money = base_money * (1 + rate_temp)
        base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
        base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
    elif buyed:
        hold_num += 1
        base = base * (1 + X1_test[i+1] / 100)
        base_money = base_money * (1 + X1_test[i+1] / 100)
        base_fee = base_fee * (1 + X1_test[i+1] / 100)
        base_money_fee = base_money_fee * (1 + X1_test[i+1] / 100)
    else:  # # 预测结果为跌 且 没有持有股票，不进行操作
        empty_num += 1
    model_line.append(base_money)
    model_line_fee.append(base_money_fee)

print('总天数:', days_test - sample_len, ',买入天数: ', buy_num, ',卖出天数: ', sell_num, ',持有天数: ', hold_num, ',空仓天数: ', empty_num)
print('最终金额: ', base_money, ' , 最终金额(含交易费): ', base_money_fee)
print("准确率: ", correct_num / (days_test - sample_len))
print("收益率: ", base - 1, " , 收益率(含交易费): ", base_fee - 1)

model_line = np.array(model_line)

fig = plt.figure()
plt.plot(dates_test[sample_len - 1:-1], close_test[sample_len - 1:-1], color='green')
plt.plot(dates_test[sample_len - 1:-1], model_line, color='red')
plt.plot(dates_test[sample_len - 1:-1], model_line_fee, color='yellow')
plt.show()
fig.savefig("pictures/model_money.jpg")

