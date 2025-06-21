import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
处理数据集
: no_nan: 去除缺失值和异常值
        input   : 待处理的datareader类型数据
        output  : 处理完的datareader类型数据

: more_feature: 处理数据，添加更多特征值
        input   : 待处理的datareader类型数据, 布尔值(是否将数据转换为窗口格式)
        output  : 处理完的datareader类型数据
'''

def no_nan(data):
    '''
    去除缺失值和异常值
    : data: 待处理的datareader类型数据
    '''
    # 检查缺失值
    print(data.isnull().sum())
    # 如果有缺失值，可以用前值填充
    data = data.ffill()

    return data

def more_feature(data, sequences=False, time_steps=10):
    '''
    处理数据, 添加更多特征值
    :param data: 待处理的 DataFrame（来自 yfinance）
    :param sequences: 是否将数据转化为窗口格式
    :param time_steps: 时间窗口大小
    :return: 处理后的训练和测试数据 (字典)
    '''
    # 计算每日收益率并生成涨跌标签
    data['Return'] = data['Close'].pct_change()
    data['Label'] = (data['Return'] > 0).astype(int).shift(-1)

    # 计算 RSI
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
        rs = gain / loss.replace(0, 1e-10)  # 用小值替代 0，避免 NaN
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # 添加特征
    data['SMA_5'] = data['Close'].rolling(window=5).mean()  # 5日均线
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20日均线
    data['RSI_14'] = calculate_rsi(data['Close'], 14)
    data.dropna(inplace=True)

    # 选择特征和目标
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'Return', 'RSI_14']
    X = data[features].values
    y = data['Label'].values

    # 数据标准化
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 按时间顺序分割数据
    train_size = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'processed_data': data  # 添加原始数据以便可视化
    }

    if sequences:
        def create_sequences(X, y, time_steps):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
        train_size_seq = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:train_size_seq]
        X_test_seq = X_seq[train_size_seq:]
        y_train_seq = y_seq[:train_size_seq]
        y_test_seq = y_seq[train_size_seq:]

        result.update({
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train_seq': y_train_seq,
            'y_test_seq': y_test_seq
        })

    print("训练集大小:", result['X_train'].shape)
    print("测试集大小:", result['X_test'].shape)
    if sequences:
        print("序列化训练集大小:", result['X_train_seq'].shape)

    return result

'''
打包
'''
def data_process_in_one(name, start_time, end_time, sequences=False, time_steps=10):
    from get_data import get_data_by_yf
    data = get_data_by_yf(name, start_time, end_time)
    data = no_nan(data)
    data = more_feature(data, sequences=sequences, time_steps=time_steps)

    return data

if __name__ == '__main__':
    from get_data import *
    data = get_data_by_yf('AAPL', '2020-01-01', '2023-01-01')
    data = no_nan(data)
    processed_data = more_feature(data, sequences=True)