import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


class HmmModel:

    # state_num = 0  # 隐状态个数
    # sample_len = 0  # 每次训练的样本长度
    # stock_name = "test" # 股票代码
    #
    # data_num = 0  # 训练传入的数据的组数
    #
    # model = None  # 高斯隐马尔可夫模型
    #
    # up_state = []  # 上涨的隐状态
    # down_state = []  # 下跌的隐状态
    # next_state = None  # 每个状态所对应的下一个状态

    def __init__(self, state_num=10, sample_len=5, stock_name="test"):
        self.state_num = state_num  # 隐状态个数
        self.sample_len = sample_len  # 每次训练的样本长度
        self.stock_name = stock_name  # 股票代码

        # self.data_num = 0  # 训练传入的数据的组数

        self.model = None  # 高斯隐马尔可夫模型

        self.up_state = []  # 上涨的隐状态
        self.down_state = []  # 下跌的隐状态
        self.next_state = None  # 每个状态所对应的下一个状态

    def train_model(self, pct_chg, *xs):
        """
            根据传入的数据训练模型
        :param pct_chg: 涨跌幅的列表
        :param xs: 训练的多组列表
        :return:
        """

        if len(xs) < 1:
            print('error: 没有传入参数XS！！！')
            return
        x_len = len(pct_chg)
        x_list = []
        for x_temp in xs:
            if len(x_temp) != x_len:
                print('error: XS参数长度不相等！！！')
                return
            x_list.append(scale(x_temp))
            # self.data_num += 1

        X = np.column_stack(x_list)  #训练集

        days = X.shape[0]  # 数据的天数

        lengths = [self.sample_len] * int(days / self.sample_len)
        if days % self.sample_len != 0:
            lengths.append(days % self.sample_len)
        lengths = np.array(lengths)

        self.model = GaussianHMM(n_components=self.state_num, n_iter=500, tol=0.001)
        self.model.fit(X, lengths)  # 训练模型————学习问题
        # print("输出根据数据训练出来的π")
        # print(self.model.startprob_)
        # print("输出根据数据训练出来的A")
        # print(self.model.transmat_)
        hidden_states = self.model.predict(X)  # 估计状态序列————解码问题，预测训练集每一天的隐状态
        # print('hidden_states len: ', len(hidden_states))

        judge_result = np.zeros(self.state_num)  # 对每个隐状态的涨跌幅求和的列表，列表的第0个元素表示隐状态0的所有交易日的涨跌幅求和的结果，以此类推

        for i in range(self.state_num):
            result_num = 0
            for temp in range(len(hidden_states)):
                if i == hidden_states[temp]:
                    judge_result[i] = judge_result[i] + pct_chg[temp]
                    result_num += 1
            judge_result[i] = judge_result[i] if result_num == 0 else judge_result[i] / result_num
        # max_down = np.max([q for q in judge_result if q < 0])  # 所有隐状态的涨跌幅求和结果中 最大的负值
        # for i in range(len(judge_result)):
        #     if judge_result[i] > 0.1:  # 隐状态的涨跌幅求和结果如果大于这个数，判断这个隐状态表示涨
        #         self.up_state.append(i)
        #     elif judge_result[i] < -0.1:  # 隐状态的涨跌幅求和结果如果小于这个数，判断这个隐状态表示跌
        #         self.down_state.append(i)
        sort_index = np.argsort(judge_result)
        span = int(self.state_num / 3)
        for i in range(span):
            self.down_state.append(sort_index[i])
        for i in range(self.state_num-span, self.state_num):
            self.up_state.append(sort_index[i])

        print(self.stock_name, "求和取均值结果: ", judge_result)
        print(self.stock_name, "涨隐状态:", self.up_state, ", 跌隐状态: ", self.down_state)

        self.next_state = np.zeros(self.state_num)  # 每个状态所对应的下一个状态的列表，第0个元素表示隐状态0的下一个隐状态，以此类推
        for i in range(self.state_num):
            self.next_state[i] = np.argmax(self.model.transmat_[i])
        print(self.stock_name, '每个状态所对应的下一个状态: ', self.next_state)

    def back_test(self, pct_chg, close, mean, *xs):
        """
            回测
        :param pct_chg: 涨跌幅列表
        :param close: 收盘价列表
        :param mean: 均价列表
        :param xs: 预测数据所需的多组列表
        :return:
        """
        dates = np.array([q for q in range(len(pct_chg))])

        if len(xs) < 1:
            print('error: 没有传入参数XS！！！')
            return
        x_len = len(pct_chg)
        if len(close) != x_len:
            print('pct_chg和close参数长度不相等！！！')
            return
        if len(mean) != x_len:
            print('pct_chg和mean参数长度不相等！！！')
            return
        x_list = []
        for x_temp in xs:
            if len(x_temp) != x_len:
                print('error: XS参数长度不相等！！！')
                return
            x_list.append(scale(x_temp))
            # self.data_num += 1

        X_test = np.column_stack(x_list)  # 测试集
        days_test = X_test.shape[0]

        correct_num = 0  # 预测正确的数量
        base_money = close[self.sample_len - 1]  # 每天的金额，以第一天的前一天的收盘价为基础金额
        base_money_fee = base_money  # 每天的金额，含手续费计算
        base = 1  # 用来计算收益率
        base_fee = base  # 用来计算含手续费的收益率
        model_line = [base_money]  # 记录每天的金额的列表
        model_line_fee = [base_money]  # 记录含手续费的每天的金额的列表
        print(self.stock_name, '初始金额 : ', base_money, ' , 实际最终金额 : ', close[days_test-1])

        buyed = 1  # 股票是否已经购买的状态
        buy_num = 0  # 购买股票的天数
        hold_num = 0  # 持有股票的天数
        sell_num = 0  # 抛出股票的天数
        empty_num = 0  # 空仓的天数

        up_num = 0      # 预测涨正确的天数
        down_num = 0    # 预测跌正确的天数
        medium_num = 0  # 预测平正确的天数

        for i in range(self.sample_len, days_test):  # 从 第"样本长度sample_len"天起到最后一天，计算金额变化。如：样本长度为5，从第六天开始计算，前五天进行隐状态预测
            arr_test = X_test[i - self.sample_len:i, :]
            states_test = self.model.predict(arr_test)  # 预测前"样本长度sample_len"天，隐状态的序列
            predict_state = self.next_state[states_test[self.sample_len - 1]]  # 预测的下一天的隐状态

            if predict_state in self.up_state and pct_chg[i] >= 1:
                up_num += 1
            elif predict_state in self.down_state and pct_chg[i] <= -1:
                down_num += 1
            elif (predict_state not in self.up_state) and (predict_state not in self.down_state) and pct_chg[i] < 1 and pct_chg[i] > -1:
                medium_num += 1

            if (predict_state in self.up_state and pct_chg[i] >= 1) or (
                    predict_state in self.down_state and pct_chg[i] <= -1) or (
                    (predict_state not in self.up_state) and (predict_state not in self.down_state) and pct_chg[i] < 1 and pct_chg[i] > -1):
                correct_num += 1    # 如果明显涨跌的隐状态对应的价格涨跌预测正确 或者 表示震荡的隐状态对应的涨跌在-1%到1%之间，则模型预测正确的次数加一

            if (predict_state not in self.down_state) and (not buyed):  # 预测结果不为跌 且 没有持有股票， 买入，手续费0.00032
                buyed = 1
                buy_num += 1
                rate_temp = (mean[i] - close[i - 1]) / close[i - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
                base = base * (1 + rate_temp)
                base_money = base_money * (1 + rate_temp)
                base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
                base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
            elif (predict_state in self.down_state) and buyed:  # 预测结果为跌 且 持有股票，抛出，手续费0.00132
                buyed = 0
                sell_num += 1
                rate_temp = (mean[i] - close[i - 1]) / close[i - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
                base = base * (1 + rate_temp)
                base_money = base_money * (1 + rate_temp)
                base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
                base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
            elif (predict_state not in self.down_state) and buyed:  # 预测结果为震荡或上涨 且 持有股票，不进行操作
                hold_num += 1
                base = base * (1 + pct_chg[i] / 100)
                base_money = base_money * (1 + pct_chg[i] / 100)
                base_fee = base_fee * (1 + pct_chg[i] / 100)
                base_money_fee = base_money_fee * (1 + pct_chg[i] / 100)
            else:  # 预测结果为跌 且 没有持有股票，不进行操作
                empty_num += 1
            model_line.append(base_money)
            model_line_fee.append(base_money_fee)

        model_line = np.array(model_line)

        print(self.stock_name, '预测涨正确的天数', up_num)
        print(self.stock_name, '预测跌正确的天数', down_num)
        print(self.stock_name, '预测平正确的天数', medium_num)

        print(self.stock_name, '总天数:', days_test - self.sample_len, ',买入天数: ', buy_num, ',卖出天数: ', sell_num, ',持有天数: ', hold_num,
              ',空仓天数: ', empty_num)
        print(self.stock_name, '最终金额: ', base_money, ' , 最终金额(含交易费): ', base_money_fee)
        print(self.stock_name, "准确率: ", correct_num / (days_test - self.sample_len))
        print(self.stock_name, "收益率: ", base - 1, " , 收益率(含交易费): ", base_fee - 1)

        ## 绘制曲线图
        fig = plt.figure()
        plt.plot(dates[self.sample_len - 1:], close[self.sample_len - 1:], color='green')
        plt.plot(dates[self.sample_len - 1:], model_line, color='red')
        plt.plot(dates[self.sample_len - 1:], model_line_fee, color='yellow')
        plt.show()
        fig.savefig("pictures\model_line_" + self.stock_name + ".jpg")
