from sklearn import datasets
import numpy as np

# 加载sk-learn load_iris
dataset = datasets.load_iris()

# X和y的shape分别为(150, 4)  (150, 1)， 这是一个3分类的数据集，y的取值为为和0，1，2
X = dataset.data
y = dataset.target
# y原本shape为(150,)，需要reshape为(150, 1)
y = np.reshape(y, (-1, 1))

class KNN:

    neighbor_points = list()

    '''
    X - 数据集
    y - 数据集label
    K - K近邻的K值
    '''
    def __init__(self, X, y, K = 5):
        self.X = X
        self.y = y
        self.K = K
    
    # （X1，X2）欧式距离
    def distance(self, X1, X2):
        return np.sqrt(np.sum(np.abs(X1 * X1 - X2 * X2), axis=1))
    
    def predict(self, X):
        y_pred = self.distance(X, self.X)
        indices = np.argsort(y_pred)[:self.K]
        class_dict = dict()
        for index in indices:
            if self.y[index][0] not in class_dict:
                class_dict[self.y[index][0]] = 1
            else:
                class_dict[self.y[index][0]] += 1

        return sorted(class_dict.items(), key=lambda X:X[1], reverse=True)[0][0]
    
    # 评估函数
    def evaluate(self, X, y):
        count = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i])
            if y_pred == y[i]:
                count = count + 1
        print("predict accuracy: %f" % ((count * 1.0) / len(X)))


model = KNN(X[:-20], y[:-20])
model.evaluate(X[-20:], y[-20:])