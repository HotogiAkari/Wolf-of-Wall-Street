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

def more_feature(data, sequences=False):
    '''
    处理数据, 添加更多特征值
    : data: 待处理的datareader类型数据
    : sequences: 确定是否将数据转化为窗口格式
    : return: 处理后的训练和测试数据(字典)
    '''
    '''
    添加特征
    '''
    # 计算 RSI
    def calculate_rsi(data, periods=14):
        '''
        计算 RSI 指标
        : data: 价格序列
        : periods: 计算周期
        : return: RSI 值
        '''
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
        rs = gain / loss.replace(0, np.nan)  # 避免除零
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # 将数据转化为时间序列窗口格式
    def create_sequences(X, y, time_steps=10):
        '''
        将数据转化为时间序列窗口格式
        : X         : 特征矩阵
        : y         : 目标向量
        : time_steps: 窗口大小
        : return    : 序列化的 X 和 y
        '''
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    # 提取 AAPL 的数据，变为单级索引
    data = data['AAPL']  

     # 添加特征
    data['SMA_5'] = data['Close'].rolling(window=5).mean()  # 5日均线
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20日均线
    data['Return'] = data['Close'].pct_change()
    data['RSI_14'] = calculate_rsi(data['Close'], 14)
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()

    '''
    选择特征和目标
    '''
    # 特征
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'Return', 'RSI_14']
    X = data[features]
    # 目标
    y = data['Target']

    '''
    数据标准化
    '''
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    '''
    按时间顺序分割数据
    '''
    # 假设用 80% 数据训练，20% 测试
    train_size = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y_scaled[:train_size]
    y_test = y_scaled[train_size:]

    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    # 如果选择转化为窗口格式
    if sequences:
        time_steps = 10
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
        train_size = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:train_size]
        X_test_seq = X_seq[train_size:]
        y_train_seq = y_seq[:train_size]
        y_test_seq = y_seq[train_size:]

        result.update({
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train_seq': y_train_seq,
            'y_test_seq': y_test_seq
        })

    # 打印信息
    print("训练集大小:", result['X_train'].shape)
    print("测试集大小:", result['X_test'].shape)
    if sequences:
        print("序列化训练集大小:", result['X_train_seq'].shape)

    return result

if __name__ == '__main__':
    from get_data import *
    data = get_data_by_yf('AAPL', '2020-01-01', '2023-01-01')
    data = no_nan(data)
    processed_data = more_feature(data, sequences=True)