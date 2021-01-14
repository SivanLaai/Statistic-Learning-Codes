from sklearn import datasets
import numpy as np
import math


def loadData(fileName):
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    #存放数据及标记
    dataArr = []; labelArr = []
    #读取文件
    fr = open(fileName)
    #遍历文件中的每一行
    for line in fr.readlines():
        #获取当前行，并按“，”切割成字段放入列表中
        #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        #split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        #将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        #在放入的同时将原先字符串形式的数据转换为0-1的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        #将标记信息放入标记集中
        #放入的同时将标记转换为整型
        #数字0标记为1  其余标记为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    #返回数据集和标记
    return dataArr, labelArr



# dataset = loadData(n_samples=300, n_features=5, n_informative=2)
# # X和y的shape分别为(300, 10)  (300, 1)， 这是一个3分类的数据集，y的取值为为和0，1
# X = dataset[0]
# # 将y转化为两个类别
# y = dataset[1]
# y = np.reshape(np.where(y == 0, -1, y), (-1, 1))




class SupportVectorMachine:

    aplha_indexs = set()

    def __init__(self, X, y, C=10.0):
        self.X = X
        self.y = y.T
        # 样本个数
        # 样本维度
        self.N, self.M = np.shape(self.X)
        print(np.shape(self.y))
        print(self.N)
        # 初始化alpha
        self.alpha = np.random.random(size=self.N) * 0.0
        self.w = np.random.random(size=self.M) * 0.0
        self.b = np.random.random() * 0.0
        self.C = C
        self.E = np.random.random(size=self.N) * 0.0
        self.Kernel = self.calcKernel()
        self.sigma = 10
    

    def calcKernel(self):
        '''
        计算核函数
        使用的是高斯核 详见“7.3.3 常用核函数” 式7.90
        :return: 高斯核矩阵
        '''
        #初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        #k[i][j] = Xi * Xj
        k = [[0 for i in range(self.N)] for j in range(self.N)]

        #大循环遍历Xi，Xi为式7.90中的x
        for i in range(self.N):
            #每100个打印一次
            #不能每次都打印，会极大拖慢程序运行速度
            #因为print是比较慢的
            if i % 100 == 0:
                print('construct the kernel:', i, self.N)
            #得到式7.90中的X
            X = self.X[i, :]
            #小循环遍历Xj，Xj为式7.90中的Z
            # 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
            # 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
            #所以小循环直接从i开始
            for j in range(i, self.N):
                #获得Z
                Z = self.X[j, :]
                #先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                #分子除以分母后去指数，得到的即为高斯核结果
                result = np.exp(-1 * result / (2 * 10**2))
                #将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        #返回高斯核矩阵
        return k

    def g(self, i):
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        gxi = 0
        for j in index:
            #计算g(xi)
            gxi += self.alpha[j] * self.y[j] * self.K(j, i)
        #求和结束后再单独加上偏置b
        gxi += self.b
        return gxi
    
    def K(self, i, j):
        return self.Kernel[i][j]
    
    def Ei(self, i):
        if self.E[i] == 0:
            self.E[i] = self.g(i) - self.y[i]
        return self.E[i]
    
    def KKTSatisfy(self, i):
        if self.alpha[i] == 0:
            return (self.y[i] * self.g(i)) >= 1
        elif self.alpha[i] == self.C:
            return (self.y[i] * self.g(i)) <= 1
        else:
            return (self.y[i] * self.g(i)) == 1
        
    
    def clip_alpha2(self, alpha2_new_unc, alpha1_old, alpha2_old, y1, y2, H, L):
        
        if alpha2_new_unc > H:
            return H
        elif alpha2_new_unc >= L:
            return alpha2_new_unc
        else:
            return L
    

    def select_alpha1(self):
        alpha2_no_distance = True
        for i in range(self.N):
            # 寻找到违反KKT条件的点j
            if self.KKTSatisfy(i) == False:
                E1 = self.Ei(i)
                E2, j = self.select_alpha2(E1, i)
                alpha1_old = self.alpha[i]
                alpha2_old = self.alpha[j]

                y1 = self.y[i]
                y2 = self.y[j]

                if y1 != y2:
                    L = max(0, alpha2_old - alpha1_old)
                    H = min(self.C, self.C + alpha2_old - alpha1_old)
                else:
                    L = max(0, alpha2_old + alpha1_old - self.C)
                    H = min(self.C, alpha2_old + alpha1_old)
                #如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                if L == H:   continue

                k11 = self.K(i, i)
                k22 = self.K(j, j)
                k21 = self.K(j, i)
                k12 = self.K(i, j)

                alpha2_new_unc = alpha2_old + (y2 * (E1 - E2) / (k11 + k22 - 2 * k12))

                alpha2_new = self.clip_alpha2(alpha2_new_unc, alpha1_old, alpha2_old, y1, y2, H, L)
                alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

                if (alpha2_new - alpha2_old) >= 10e-4:
                    alpha1_distance = False

                # print("alpha2_new: ", alpha2_new)

                b1_new = -E1 - y1 * k11 * (alpha1_new - alpha1_old) - y2 * k21 * (alpha2_new - alpha2_old) + self.b
                b2_new = -E2 - y2 * k12 * (alpha1_new - alpha1_old) - y2 * k22 * (alpha2_new - alpha2_old) + self.b

                b_new = self.select_b_new(b1_new, b2_new, alpha1_new, alpha2_new)

                

                self.alpha[i] = alpha1_new
                self.alpha[j] = alpha2_new
                self.b = b_new

                self.E[i] = self.Ei(i)
                self.E[j] = self.Ei(j)

                
        return alpha2_no_distance

    def select_b_new(self, b1_new, b2_new, alpha1_new, alpha2_new):
        if alpha1_new > 0 and alpha1_new < self.C and alpha2_new > 0 and alpha2_new < self.C:
            return b1_new
        else:
            return (b1_new + b2_new) / 2.0
    
    def select_alpha2(self, E1, i):
        alpha2_index = -1
        # max|E1 - E2|
        max_abs = -1
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        E2 = 0
        for j in nozeroE:
            E2_tmp = self.Ei(j)
            if np.abs(E1 - E2_tmp) > max_abs:
                max_abs = math.fabs(E1 - E2_tmp)
                alpha2_index = j
                E2 = E2_tmp
        
        if alpha2_index == -1:
            alpha2_index = i
            while alpha2_index == i:
                #获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                alpha2_index = int(random.uniform(0, self.N))
            #获得E2
            E2 = self.Ei(maxIndex)
        return E2, alpha2_index

            


    '''
    iter - 迭代次数，取值为整数
    lr - 学习率，取值为(0, 1]之间的浮点数
    '''
    def train(self, iter = 100):
        for i in range(iter):
            # 寻找alpha1
            if not self.select_alpha1():
                break
            
    
    
    # sign函数
    def sign(self, X):
        return np.where(X >= 0.0, 1, 0)
    
    def kernel(self, x1, x2):
        #按照“7.3.3 常用核函数”式7.90计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * 10 ** 2))
        #返回结果
        return np.exp(result)
    
    def predict(self, x):
        value = 0
        for i in range(self.N):
            value += self.alpha[i] * self.y[i] * self.kernel(self.X[i, :], x)
        value += self.b
        return np.sign(value)
    
    # 评估函数
    def evaluate(self, X, y):
        y_pred = []
        y_real = list()
        for i in range(len(X)):
            y_tmp = int(self.predict(X[i]))       
            y_pred.append(y_tmp)
            y_real.append(int(y[i]))
        
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_real[i]:
                count += 1
        acc = (count * 1.0) / len(y_pred)
        print("predict accuracy: %f" % acc)


X, y = loadData('./mnist_train.csv')


X = np.mat(X[:1000])
y = np.mat(y[:1000])

model = SupportVectorMachine(X, y)
model.train()
y = y.T
model.evaluate(X[-20:], y[-20:])