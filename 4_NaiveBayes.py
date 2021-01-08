from sklearn import datasets
import numpy as np

# 加载sk-learn breast_cancer公共数据集
dataset = datasets.load_breast_cancer()

# X和y的shape分别为(569, 30) (569, 1)，y的取值为为和1，1表示正样本，0表示负样本
X = dataset.data
y = dataset.target

class NaiveBayes:

    '''
    X - 数据集
    y - 数据集label
    lamda - 拉普拉斯平滑
    '''
    def __init__(self, X, y, lamda = 1.0):
        self.X = X
        self.y = y
        self.N = len(X)
        # 样本维度
        self.D = len(X[-1])
        # 样本分类
        self.class_set = set(y)
        # 拉普拉斯平滑
        self.lamda = lamda
    
    
    def predict(self, x):
        # 类k的计算
        pred = None
        prob = 0.0
        for curr_class in self.class_set:
            # 计算第K个类别的概率估计
            Ck_count = np.sum(np.where(y == curr_class, 1, 0))
            p_CK = Ck_count / (self.N * 1.0)
            # 计算每个特征的概率估计，这里有30个特征
            for j in range(self.D):
                p_Aj_Ck = p_CK
                # 计算第j个特征在第l和取值上的概率
                feature_value_set = set(self.X[:, j])
                # 得到当前的Sj值
                Sj = len(feature_value_set)
                for l in range(Sj):
                    curr_feature_value = list(feature_value_set)[l]
                    # 求取P(Xj = ajl | Y = ck)
                    Ajl_Ck_count = np.sum(np.where(self.X[:, j] == x[j], 1, 0)) + self.lamda
                    # 连乘P(Xj = ajl | Y = ck)求P(Xj = xj | Y = ck)，因为一直乘以概率会接近0，所以这里用对数求和来等价求解
                    p_Ajl_Ck = Ajl_Ck_count / (Ck_count + Sj * self.lamda)
                    p_Aj_Ck = p_Aj_Ck + p_Ajl_Ck
            if prob < p_Aj_Ck:
                pred = curr_class
                prob = p_Aj_Ck
        return pred

    
    # 评估函数
    def evaluate(self, X, y):
        count = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i])
            if y_pred == y[i]:
                count = count + 1
        print("predict accuracy: %f" % ((count * 1.0) / len(X)))


model = NaiveBayes(X[:-20], y[:-20])
# model.predict(X[0])
# print(y[0])
model.evaluate(X[-20:], y[-20:])