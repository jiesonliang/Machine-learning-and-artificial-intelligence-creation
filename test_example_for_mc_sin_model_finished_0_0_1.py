import pandas as pd
import numpy as np
from mc_sin_model_finished_0_0_1 import MultiSinModel

# 获取股票历史数据
def load_stock_data(stock_code, start_date, end_date):
    # 这里是获取股票数据的逻辑，例如使用tushare、yfinance等库
    # 为了简化演示，这里假设返回的是一个包含'Open', 'High', 'Low', 'Close', 'Volume'等列的DataFrame
    stock_data = pd.read_csv(f'{stock_code}')  # 示例：从CSV文件加载数据
    return stock_data[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)]

# 获取特征和目标变量
def prepare_stock_data(stock_data):
    # 示例：将'Open'和'High'作为特征，'Close'作为目标变量
    x_data = stock_data[['open', 'high','low','volume','price_change','p_change']].values
    y_data = stock_data['close'].values
    return x_data, y_data

# 加载股票代码为'stock_code'的指定日期范围内的数据
stock_code = '601888'  # 示例股票代码（深交所）
start_date = '2021-03-30'
end_date = '2023-09-28'

stock_data = load_stock_data(stock_code, start_date, end_date)
x_data, y_data = prepare_stock_data(stock_data)

# 训练模型并打印损失
model = MultiSinModel(n_features=6, n_t_params=6)
losses = model.optimize_with_loss_printing(x_data, y_data)

# 保存模型
model.save_model('multi_sin_model.pkl')

# 加载模型
loaded_model = MultiSinModel.load_model('multi_sin_model.pkl')

# 测试数据
test_x = np.array([[180.15,108.15,105.51,164757.44,-1.53,-1.42]])
test_y = np.array([105.97])

# 使用加载的模型进行预测
predictions = model.evaluate(test_x,test_y)

# 打印输出
print('Predictions:', predictions)
print('Expected output:', test_y)


