import numpy as np
import pickle
import random as rd


class MultiSinModel:
    """
    多余弦模型类，用于拟合包含多个特征的余弦函数数据。

    参数:
    n_features(int): 特征数量。
    n_t_params(int): 时间参数数量。

    属性:
    coefficients(numpy.ndarray): 模型系数数组，包括特征参数和时间参数。
    """

    def __init__(self, n_features, n_t_params):
        """
        初始化多余弦模型。

        参数:
        n_features(int): 特征数量。
        n_t_params(int): 时间参数数量。
        """
        self.n_features = n_features
        self.n_t_params = n_t_params
        self.coefficients = np.array([rd.randint(-100,100)+rd.random() for i in range(n_features * 3 + n_t_params)])#np.ones(n_features * 3 + n_t_params)*80
        #参数设定影响模型loss

    def evaluate(self, x_data, y_data, t_params=None):
        """
        根据输入数据和参数评估模型输出。

        参数:
        x_data(numpy.ndarray): 输入特征数据。
        y_data(numpy.ndarray): 目标数据。
        t_params(numpy.ndarray, 可选): 时间参数。如果未提供，则使用模型的系数中的时间参数。

        返回:
        numpy.ndarray: 模型预测的目标值。
        """
        if t_params is None:
            t_params = self.coefficients[self.n_features * 3: self.n_features * 3 + self.n_t_params]

        a_values = self.coefficients[:self.n_features * 3:3]
        c_values = self.coefficients[1:self.n_features * 3:3]
        b_values = self.coefficients[2:self.n_features * 3:3]

        y = self.coefficients[-1] * np.ones_like(y_data)

        for i in range(self.n_features):
            b_i = t_params[i] + x_data[:, i]
            y += a_values[i] * np.sin(c_values[i] * b_i)

        return y

    def gradient_eval(self, x_data, y_data, t_params=None):
        """
        计算模型系数的梯度。

        参数:
        x_data(numpy.ndarray): 输入特征数据。
        y_data(numpy.ndarray): 目标数据。
        t_params(numpy.ndarray, 可选): 时间参数。如果未提供，则使用模型的系数中的时间参数。

        返回:
        numpy.ndarray: 模型系数的梯度数组。
        """
        if t_params is None:
            t_params = self.coefficients[self.n_features * 3: self.n_features * 3 + self.n_t_params]

        gradients = np.zeros_like(self.coefficients)
        a_values = self.coefficients[:self.n_features * 3:3]
        c_values = self.coefficients[1:self.n_features * 3:3]
        b_values = self.coefficients[2:self.n_features * 3:3]

        for i in range(self.n_features):
            sin_value = np.sin(c_values[i] * (t_params[i] + x_data[:, i]))
            cos_value = np.cos(c_values[i] * (t_params[i] + x_data[:, i]))

            gradients[i * 3] = np.mean(sin_value * (y_data - self.evaluate(x_data, y_data, t_params)))
            gradients[i * 3 + 1] = np.mean(
                a_values[i] * b_values[i] * cos_value * (y_data - self.evaluate(x_data, y_data, t_params)))
            gradients[i * 3 + 2] = np.mean(
                a_values[i] * c_values[i] * cos_value * (y_data - self.evaluate(x_data, y_data, t_params)))

            gradients[self.n_features * 3 + i] = np.mean(
                -a_values[i] * c_values[i] * cos_value * (y_data - self.evaluate(x_data, y_data, t_params)))

        gradients[-1] = np.mean((y_data - self.evaluate(x_data, y_data, t_params)))

        return gradients

    def optimize(self, x_data, y_data, learning_rate=0.00001, num_iterations=20):
        """
        使用梯度下降法优化模型系数。

        参数:
        x_data(numpy.ndarray): 输入特征数据。
        y_data(numpy.ndarray): 目标数据。
        learning_rate(float, 可选): 学习率，默认为0.001。
        num_iterations(int, 可选): 迭代次数，默认为10。
        """
        for _ in range(num_iterations):
            gradients = self.gradient_eval(x_data, y_data)
            self.coefficients -= learning_rate * gradients

    def loss(self, x_data, y_data):
        """
        计算模型的损失函数。

        参数:
        x_data(numpy.ndarray): 输入特征数据。
        y_data(numpy.ndarray): 目标数据。

        返回:
        float: 模型的损失值。
        """

        def mean_squared_error(y_true, y_pred):
            return np.mean(((y_true - y_pred) ** 2) / 2)

        y_pred = self.evaluate(x_data, y_data)
        return mean_squared_error(y_data, y_pred)

    def optimize_with_loss_printing(self, x_data, y_data, learning_rate=0.00005, num_iterations=800):
        """
        使用梯度下降法优化模型系数，并打印每次迭代的损失值。

        参数:
        x_data(numpy.ndarray): 输入特征数据。
        y_data(numpy.ndarray): 目标数据。
        learning_rate(float, 可选): 学习率，默认为0.00001。
        num_iterations(int, 可选): 迭代次数，默认为300。

        返回:
        list: 每次迭代的损失值列表。
        """
        losses = []
        for i in range(num_iterations):
            gradients = self.gradient_eval(x_data, y_data)
            self.coefficients -= learning_rate * gradients
            if i % 10 == 0:
                loss = self.loss(x_data, y_data)
                print(f"Iteration {i}: Loss is {loss:.4f}")
                losses.append(loss)

        return losses

    def save_model(self, filename):
        """
        保存模型至文件。

        参数:
        filename(str): 文件名。
        """
        model_dict = {'coefficients': self.coefficients,
                      'n_features': self.n_features,
                      'n_t_params': self.n_t_params}
        with open(filename, "wb") as f:
            pickle.dump(model_dict, f)

    @classmethod
    def load_model(cls, filename):
        """
        从文件加载模型。

        参数:
        filename(str): 文件名。

        返回:
        MultiSinModel: 加载的模型实例。
        """
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)
            loaded_model = cls(model_dict['n_features'], model_dict['n_t_params'])
            loaded_model.coefficients = model_dict['coefficients']
            return loaded_model


# 使用示例：
# 假设数据集已经预处理好并赋值给x_data和y_data，且它们的形状分别为 (n_samples, n_features) 和 (n_samples,)
#model = MultiSinModel(n_features=2, n_t_params=2)
#model.optimize(x_data, y_data)

'''
MultiSinModel 类实现了一个基于多个正弦函数的模型，该模型可以用于拟合具有周期性特征的数据集。在具体应用场景中，这样的模型可能适用于：
时间序列预测：当数据呈现出多个不同频率或相位的周期波动时，例如电力负荷预测、气温变化、经济指标波动等。
物理学和工程领域：在分析涉及多个正弦波叠加现象的问题时，如声波、光波、振动分析等。
生物医学信号处理：对心电图（ECG）、脑电图（EEG）等生理信号进行建模，这些信号通常包含多个频率成分。
数据挖掘与机器学习：在某些复杂系统的建模中，如果发现系统行为可以通过多个调制的正弦波来描述，那么这个模型就可以用来作为非线性回归或者分类任务的一部分。
根据代码中的实现细节，模型可以根据输入特征 x_data 通过调整系数计算输出 y，并且支持梯度下降优化，这表明它可以被训练以最小化预测值与真实目标值之间的差异，例如均方误差损失。此外，模型还提供了保存和加载的功能，方便模型训练后结果的应用和复用。
'''

'''
y=asin(bc1)+a2sin(b2c2)……+f，c1=t1+x1,c2=t2+x2
您提供的模型似乎是一个基于多个正弦函数的叠加模型，其中每个正弦函数具有不同的幅度（a1, a2, ..., f）、频率（b1, b2, ...）以及时间-特征组合后的输入变量（c1 = t1 + x1, c2 = t2 + x2）。用数学公式表示这个模型的大致形式为：
[ y = \sin(b_1(c_1)) \cdot a_1 + \sin(b_2(c_2)) \cdot a_2 + \dots + f ]
其中：
y 是模型的输出。
asin(bc) 表示幅度为 a、频率参数为 b 且输入变量为 c 的正弦函数。
t1, t2, ... 是时间相关的变量。
x1, x2, ... 是与时间无关但会影响正弦函数输入的特征变量。
请注意，通常在实际应用中，这样的模型可能还需要包含一个学习过程来确定最佳的幅度和频率参数。同时，上述表达式没有明确说明是否所有的时间和特征变量都会参与到每一个正弦函数中，这取决于您的具体实现方式。如果每个正弦函数都对应一个特定的时间和特征组合，则模型应该有相应匹配的输入项。
'''

'''
MultiSinModel 类实现的模型具有以下特点和可能的优势：
特定任务高效：由于该模型是基于多个正弦函数构建的，对于那些可以有效用多个调制或未调制正弦波形表示的数据集，它可能会比通用机器学习算法或神经网络在拟合此类数据时更直接、更高效。特别是当目标函数本身具有明确的周期性结构时，这个模型可以直接捕捉这种周期性特征。
计算效率与可解释性：相比于复杂的神经网络结构，MultiSinModel 可能拥有较低的计算复杂度，特别是在训练阶段。此外，模型参数（如频率、幅度、相位以及时间参数）具有直观的物理意义，这使得模型输出更容易解释，有助于提升结果的可理解性。
较小的参数量：相比于深度神经网络，该模型通常拥有较少的参数数量，这可能减少过拟合的风险，并且对数据量的需求相对较小。
优化路径清晰：由于损失函数相对简单，梯度计算也更为直接，因此在优化过程中，模型收敛路径可能更加稳定和可控。
然而，相较于神经网络或其他高级机器学习算法，MultiSinModel 的劣势在于其假设数据具有某种特殊的周期性结构，这意味着它不适用于非线性、非周期性的复杂数据建模。同时，神经网络能够自动学习并提取数据中的任意复杂特征，而 MultiSinModel 无法做到这一点，它的泛化能力受限于预定义的函数形式。
总结来说，在处理包含明显周期性成分的问题上，MultiSinModel 能够提供一种简洁明了的解决方案，但在处理更广泛、更复杂的机器学习问题时，可能不如神经网络等方法灵活和强大。
'''