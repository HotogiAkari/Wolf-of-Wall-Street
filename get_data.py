import yfinance as yf

'''
获取数据
: get_data_by_yf: 从yahoo获取股票数据
        input   : 股票代码, 开始时间, 截止时间
        output  : 股票数据集, 数据类型为多级索引(MultiIndex)
'''

def get_data_by_yf(name, start_time, end_time):
    '''
    使用yfinance获取数据
    : name       : 股票名
    : start_time : 开始时间
    : end_time   : 结束时间
    return: 返回pandas DataFrame对象类型
    '''
    try:
        data = yf.download(name, start=start_time, end=end_time)
        if data.empty:
            raise ValueError(f"找不到股票代码 {name}")
        return data
    except Exception as e:
        print(f"get_data_by_yf函数发生错误: {e}")
        return None

if __name__ == '__main__':
    data_yf = get_data_by_yf('AAPL', '2020-01-01', '2023-01-01')
    print(data_yf)