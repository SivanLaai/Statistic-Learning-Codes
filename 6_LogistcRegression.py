from sklearn import datasets
import numpy as np

# 加载sk-learn breast_cancer公共数据集
dataset = datasets.load_breast_cancer()

# X和y的shape分别为(569, 30) (569, 1)，y的取值为为和1，1表示正样本，0表示负样本
X = dataset.data
y = dataset.target
# y原本shape为(569,)，需要reshape为(569, 1)
y = np.reshape(y, (-1, 1))

class LogistcRegression:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        # 样本个数
        self.N = len(X)
        # 样本维度
        self.M = len(X[-1])
        # 随机生成感知机的参数w和偏置b
        self.w = np.random.random(size=self.M) * 0.0
        self.b = np.random.random() * 0.0

    def calc_loss(self):
        z = (np.dot(self.X, self.w.T) + self.b)
        loss = -np.sum(self.y * np.log(self.sigmoid(z)) + (1 - self.y) * np.log(1 - self.sigmoid(z))) / self.N
        return loss

    def calc_b_grad(self):
        z = (np.dot(self.X, self.w.T) + self.b)
        z = self.sigmoid(z)
        b_grad = -np.sum((self.y - z))  / self.N
        return b_grad

    def calc_w_grad(self):
        z = (np.dot(self.X, self.w.T) + self.b)
        z = np.reshape(self.sigmoid(z), (-1, 1))
        w_grad = -np.sum((self.y  - z) * self.X, axis=0) / self.N
        return w_grad

    
    # 梯度更新w
    def update_w(self, lr):
        self.w = self.w - lr * self.calc_w_grad()
    
    # 梯度更新b
    def update_b(self, lr):
        self.b = self.b - lr * self.calc_b_grad()


    '''
    iter - 迭代次数，取值为整数
    lr - 学习率，取值为(0, 1]之间的浮点数
    '''
    def train(self, iter = 1000, lr = 0.00003):
        for i in range(iter):
            self.update_w(lr)
            self.update_b(lr)
            loss = self.calc_loss()
            print(i, ": loss = ", loss)
    
    def sigmoid(self, x):
        z = 1 / (1 + np.exp(-x))
        # print(z)
        return z
    
    # sign函数
    def sign(self, X):
        return np.where(self.sigmoid(X) >= 0.5, 1, 0)
    
    def predict(self, X):
        y_pred = self.sign(np.dot(X, self.w.T) + self.b)
        return y_pred
    
    # 评估函数
    def evaluate(self, X, y):
        y_pred = np.reshape(self.predict(X), (-1, 1))
        acc = (np.sum(np.where(y_pred==y, 1, 0)) * 1.0) / len(y_pred)
        print("predict accuracy: %f" % acc)

model = LogistcRegression(X[:-20], y[:-20])
model.train()
model.evaluate(X[-20:], y[-20:])