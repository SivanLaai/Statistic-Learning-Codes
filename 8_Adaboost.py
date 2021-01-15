from sklearn import datasets
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

# 加载sk-learn breast_cancer公共数据集
dataset = datasets.load_breast_cancer()

# X和y的shape分别为(569, 30) (569, 1)，y的取值为为和1，1表示正样本，0表示负样本
X = dataset.data
y = dataset.target
# y原本shape为(569,)，需要reshape为(569, 1)
y = np.reshape(np.where(y == 0, -1, y), (-1, 1))

class Adaboost:

    # Gm(xi) != yi的所有样本
    G_f_index = set()

    #
    def __init__(self, X, y, base_model="decision_tree"):
        self.X = X
        self.y = np.reshape(y, (-1, 1))
        print(np.shape(self.y))
        # 样本个数
        self.N = len(X)
        # 样本维度
        self.M = len(X[-1])
        # 每个模型的权重，随着训练，动态增加
        self.a = list()
        # 每个样本的权重，初始化为1/N
        self.W = np.ones((self.N, 1)) / self.N
        self.base_model = base_model
        self.G = list()
    
    # 计算em，这里只需要知道Gm(xi) != yi的所有样本，然后叠加权重就可以
    def calc_em(self):
        indices = list(self.G_f_index)
        em = np.sum(self.W[indices]) + 10e-6
        return em

    # 计算am的值
    def calc_am(self):
        em = self.calc_em()
        am = math.log((1 - em) / em) / 2.0
        
        return am
    
    def calc_Zm(self, am, Gm, y):
        Zm = np.sum(self.W * np.exp(-am * y * Gm))
        return Zm
    
    def upate_G_f_index(self, m):
        Gm = np.reshape(self.G[m].predict(self.X), (-1, 1))
        y = self.y
        self.G_f_index = set(np.argwhere(Gm != y)[:, 0])
    
    # 更新wm的值
    def update_wm(self, m):
        am = self.calc_am()
        Gm = self.G[m].predict(self.X)
        y = self.y
        Zm = self.calc_Zm(am, Gm, y)
        self.W = (self.W  / Zm) * np.exp(-am * y * Gm)
    
    def train_model_m(self, m):
        # 默认的模型就是决策树桩
        curr_model = DecisionTreeClassifier(max_depth=1)
        curr_model.fit(self.X, self.y)
        self.G.append(curr_model)
        self.upate_G_f_index(m)
        am = self.calc_am()
        self.a.append(am)



    '''
    iter - 迭代次数，取值为整数
    '''
    def train(self, iter = 5):
        for i in range(iter):
            self.train_model_m(i)
            self.update_wm(i)
    
    # sign函数
    def sign(self, X):
        return np.where(X < 0, -1, 1)
    
    def fx(self, x):
        Gx_val = []
        x = np.reshape(x, (1, -1))
        for Gx in self.G:
            Gx_val.append(Gx.predict(x))
        y_pred = self.sign(np.sum(np.array(Gx_val) * np.array(self.a)))
        return y_pred
    
    # 评估函数
    def evaluate(self, X, y):
        y_pred_list = list()
        for i in range(len(X)):
            y_pred_list.append(self.fx(X[i]))
        y_pred_list = np.reshape(y_pred_list, (-1, 1))
        y = np.reshape(y, (-1, 1))
        acc = (np.sum(np.where(y_pred_list==y, 1, 0)) * 1.0) / len(y_pred_list)
        print("predict accuracy: %f" % acc)

model = Adaboost(X[:-50], y[:-50])
print("before train:")
model.evaluate(X[-50:], y[-50:])
model.train()
print("after train:")
model.evaluate(X[-50:], y[-50:])