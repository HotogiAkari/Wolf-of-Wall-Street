import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.*models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

from data_process import data_process_in_one

# 检查 TensorFlow 和 GPU
print("TensorFlow 版本:", tf.__version__)
print("GPU 可用:", tf.config.list_physical_devices('GPU'))

# 构建改进的 LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 主程序
if __name__ == "__main__":
    # 参数设置
    ticker = "AAPL"
    start_date = "2000-01-01"
    end_date = "2025-01-01"
    time_steps = 20  # 调整时间窗口

    # 数据预处理并生成序列
    data_dict = data_process_in_one(ticker, start_date, end_date, sequences=True, time_steps=time_steps)

    # 提取序列化数据并转换为张量
    X_train = tf.convert_to_tensor(data_dict['X_train_seq'], dtype=tf.float32)
    X_test = tf.convert_to_tensor(data_dict['X_test_seq'], dtype=tf.float32)
    y_train = tf.convert_to_tensor(data_dict['y_train_seq'], dtype=tf.float32)
    y_test = tf.convert_to_tensor(data_dict['y_test_seq'], dtype=tf.float32)

    print("X_train 形状:", X_train.shape)
    print("y_train 形状:", y_train.shape)
    print("训练集标签分布:", np.bincount(y_train.numpy().astype(int)))
    print("测试集标签分布:", np.bincount(y_test.numpy().astype(int)))

    # 构建并训练模型
    model = build_lstm_model(input_shape=(time_steps, X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy:.4f}")

    # 预测
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 打印预测结果和性能指标
    y_test_np = y_test.numpy().flatten()
    y_pred_binary_np = y_pred_binary.flatten()
    print("预测结果 (y_pred_binary):", y_pred_binary_np)
    print("实际结果 (y_test):", y_test_np)
    cm = confusion_matrix(y_test_np, y_pred_binary_np)
    print("混淆矩阵:")
    print(cm)
    print("分类报告:")
    print(classification_report(y_test_np, y_pred_binary_np, target_names=['下跌', '上涨']))

    # 可视化结果
    test_dates = data_dict['processed_data'].index[-len(y_test):]
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, data_dict['processed_data']['Close'][-len(y_test):], label='收盘价')
    plt.scatter(test_dates, y_test_np, color='green', label='实际涨跌')
    plt.scatter(test_dates, y_pred_binary_np, color='red', label='预测涨跌')
    plt.legend()
    plt.title(f"{ticker} 股票涨跌预测")
    plt.show()

    # 可视化训练过程
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.legend()
    plt.title('模型准确率')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.legend()
    plt.title('模型损失')
    plt.show()