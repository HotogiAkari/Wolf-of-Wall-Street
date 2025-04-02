import os
import pickle

'''
保存获取的数据集或模型
: save_data: 保存处理后的数据集为csv和pkl文件
        input   : 处理后的数据 (路径和文件名)
'''

def save_data(processed_data, directory=None, csv_file='processed_data.csv', pickle_file='processed_data.pkl'):
    '''
    将处理后的数据保存为 CSV 和 Pickle 文件
    : processed_data: 处理后的数据字典
    : directory: 保存目录，默认为 './data'
    : csv_file: CSV 文件名
    : pickle_file: Pickle 文件名
    '''
    # 如果未指定目录，默认使用脚本所在目录下的 'data' 文件夹
    if directory is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
        directory = os.path.join(script_dir, 'data')  # 构造 data 子目录路径

    os.makedirs(directory, exist_ok=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'Return', 'RSI_14']

    # 保存为 CSV
    csv_data = []
    df_train = pd.DataFrame(processed_data['X_train'], columns=features)
    df_train['Target'] = processed_data['y_train']
    df_train['Type'] = 'train'
    csv_data.append(df_train)

    df_test = pd.DataFrame(processed_data['X_test'], columns=features)
    df_test['Target'] = processed_data['y_test']
    df_test['Type'] = 'test'
    csv_data.append(df_test)

    if 'X_train_seq' in processed_data:
        for i in range(processed_data['X_train_seq'].shape[0]):
            df_seq = pd.DataFrame(processed_data['X_train_seq'][i], columns=features)
            df_seq['Target'] = processed_data['y_train_seq'][i]
            df_seq['Type'] = f'train_seq_{i}'
            csv_data.append(df_seq)
        for i in range(processed_data['X_test_seq'].shape[0]):
            df_seq = pd.DataFrame(processed_data['X_test_seq'][i], columns=features)
            df_seq['Target'] = processed_data['y_test_seq'][i]
            df_seq['Type'] = f'test_seq_{i}'
            csv_data.append(df_seq)

    combined_df = pd.concat(csv_data, ignore_index=True)
    csv_path = os.path.join(directory, csv_file)
    combined_df.to_csv(csv_path, index=False)
    print(f"CSV 文件已保存到 {csv_path}")

    # 保存为 Pickle
    pickle_path = os.path.join(directory, pickle_file)
    with open(pickle_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Pickle 文件已保存到 {pickle_path}")

if __name__ == '__main__':
    from get_data import get_data_by_yf
    from data_process import *
    data = get_data_by_yf('AAPL', '2020-01-01', '2023-01-01')
    data = no_nan(data)
    processed_data = more_feature(data, sequences=True)
    save_data(processed_data, directory='dataset', csv_file='processed_data.csv', pickle_file='processed_data.pkl')