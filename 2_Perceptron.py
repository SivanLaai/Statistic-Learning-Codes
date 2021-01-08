from sklearn import datasets
import numpy as np

# 加载sk-learn breast_cancer公共数据集
dataset = datasets.load_breast_cancer()

# X和y的shape分别为(569, 30) (569, 1)，y的取值为为和1，1表示正样本，0表示负样本
X = dataset.data
y = dataset.target
# y原本shape为(569,)，需要reshape为(569, 1)
y = np.reshape(np.where(y == 0, -1, y), (-1, 1))

class Perceptron:

    err_index_set = set()

    def __init__(self, X, y):
        self.X = X
        self.y = y
        # 样本个数
        self.N = len(X)
        # 样本维度
        self.M = len(X[-1])
        # 随机生成感知机的参数w和偏置b
        self.w = np.random.randn(1, self.M)
        self.b = np.random.randn()
    
    # 获取样本的分类错误的下标
    def get_err_index(self):
        self.err_index_set.clear()
        dot = y * self.sign(np.dot(X, self.w.T) + self.b)
        self.err_index_set.update(np.argwhere(dot <= 0)[:, 0])

    
    # 梯度更新w
    def update_w(self, lr):
        set_index = np.random.randint(len(self.err_index_set)) - 1
        index = list(self.err_index_set)[set_index]
        self.w = self.w + lr * X[index] * y[index]
    
    # 梯度更新b
    def update_b(self, lr):
        set_index = np.random.randint(len(self.err_index_set)) - 1
        index = list(self.err_index_set)[set_index]
        self.b = self.b + lr * y[index]


    '''
    iter - 迭代次数，取值为整数
    lr - 学习率，取值为(0, 1]之间的浮点数
    '''
    def train(self, iter = 10000, lr = 0.75):
        for i in range(iter):
            self.get_err_index()
            # 如果没有错误分类点，直接退出
            if len(self.err_index_set) == 0:
                break
            self.update_w(lr)
            self.update_b(lr)
    
    # sign函数
    def sign(self, X):
        return np.where(X < 0, -1, 1)
    
    def predict(self, X):
        y_pred = self.sign(np.dot(X, self.w.T) + self.b)
        return y_pred
    
    # 评估函数
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = (np.sum(np.where(y_pred==y, 1, 0)) * 1.0) / len(y_pred)
        print("predict accuracy: %f" % acc)

model = Perceptron(X[:-20], y[:-20])
print("before train:")
model.evaluate(X[-20:], y[-20:])
model.train()
print("after train:")
model.evaluate(X[-20:], y[-20:])