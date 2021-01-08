from sklearn import datasets
import numpy as np

# 加载sk-learn breast_cancer公共数据集
dataset = datasets.make_classification(n_samples=300, n_features=5, n_informative=2)
# X和y的shape分别为(300, 10)  (300, 1)， 这是一个3分类的数据集，y的取值为为和0，1
X = dataset[0]
# 将y转化为两个类别
y = dataset[1]

# C4.5决策分类树，
class DecisionTreeClassifier:

    root_tree = dict()
    used_index = set()

    '''
    X - 数据集
    y - 数据集label
    '''
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = len(X)
        # 样本维度
        self.D = len(X[-1])
        # 样本分类
        self.class_set = set(y)
    
    '''
    计算信息增益
    y - 标签
    '''
    def calc_gainD(self, y):
        gainD = 0.0
        for curr_c in set(y):
            Ck_count = list(y).count(curr_c)
            gainD += -((Ck_count / len(y))) * np.log2(Ck_count / len(y))
        return gainD
    
    '''
    计算信息增益比
    X - 数据集
    index - 特征列的下标
    '''
    def calc_gainDARate(self, X, y, index):
        gainDA = 0.0
        sorted_points = sorted(X[:, index])
        # 连续型的需要求出两两之间的中值点，也就是看成了离散的形式
        divide_values = [(sorted_points[i] + sorted_points[i - 1]) / 2 for i in range(1, len(sorted_points))]
        divide_value = 0.0
        gain_HA = 0.0
        for curr_value in divide_values:
            DA = np.where(X[:, index] <= curr_value, 1, 0)
            D1 = y[np.argwhere(DA == 1)[:, 0]]
            D2 = y[np.argwhere(DA == 0)[:, 0]]
            gain_HA += self.calc_gainD(DA)
            curr_gainDA = self.calc_gainD(y) - (( float(len(D1)) / len(y) ) * self.calc_gainD(D1)) - (( float(len(D2)) / len(y) ) * self.calc_gainD(D2))
            if curr_gainDA > gainDA:
                gainDA = curr_gainDA
                divide_value = curr_value
        gainDARate = gainDA / gain_HA
        return gainDARate, divide_value
    

    def getOptimalIndex(self, X, y):
        optimal_index = 0
        optimal_value = 0
        GainDARate = 0.0
        for i in range(self.D):
            if i in self.used_index:
                continue
            curr_GainDARate, curr_value = self.calc_gainDARate(X, y, i)
            if curr_GainDARate > GainDARate:
                optimal_index = i
                GainDARate = curr_GainDARate
                optimal_value = curr_value
        if optimal_index in self.used_index:
            return -1, -1
        self.used_index.add(optimal_index)
        return optimal_index, optimal_value

    

    # 生成决策树
    def train(self):
        self.root_tree = self.generateTree(self.X, self.y)
        print(self.root_tree)
    
    # 得到类别个数最多的那个类，因为这里只是二分类
    def getMajority(self, y):
        # 类别1个数的两倍大于总个数，则返回1，否则返回0
        if len(y) < (np.sum(np.where(y == 1, 1, 0)) * 2):
            return 1
        return 0
    

    # 生成决策树
    def generateTree(self, X, y):
        tree = dict()
        # 只有一个类别的时候，直接就返回这个类别
        if len(set(y)) == 1:
            tree = y[-1]
            return tree

        optimal_index, optimal_value = self.getOptimalIndex(X, y)
        if optimal_index == -1:
            tree = self.getMajority(y)
            return tree
        D1_index = np.argwhere(X <= optimal_value)[:, 0]
        D2_index = np.argwhere(X > optimal_value)[:, 0]
        # X[D1_index], y[D1_index], D2_index], y[D2_index]
        tree = {optimal_index : {"<= "+str(optimal_value) : self.generateTree(X[D1_index], y[D1_index]), "> "+str(optimal_value) : self.generateTree(X[D2_index], y[D2_index])}}  
        return tree
        
    '''        
    解析决策树，这里用到了递归，字典格式如下 {2: {'<= 1.9': {3: {'<= 0.6': 1, '> 0.6': {0: {'<= 5.4': {1: {'<= 2.9': 1, '> 2.9': 1}}, '> 5.4': 0}}}}, '> 1.9': 0}}
    {1: {'<= 2.9': 1, '> 2.9': 1}}, 的含义是，在特征列为1，值小于等于2.9的时候为类别1，大于1的时候也为类别1，如果会也为字典则需要继续递归。
    x - 输入数据，为了简单，这里是一个样本，长度即为特征个数
    node - 生成的决策树的根节点
    '''
    def getValue(self, x, node):
        if type(node) != dict:
            # print(node)
            return node
        for key in node:
            col_index = key
            curr_value = x[col_index]
            for divide in node[key]:
                divide_value = float(divide.split(' ')[-1])
                if "<=" in divide and curr_value <= divide_value:
                    return self.getValue(x, node[key][divide])
                if ">" in divide and curr_value > divide_value:
                    return self.getValue(x, node[key][divide]) 


    
    def predict(self, x):
        # 类k的计算
        return self.getValue(x, self.root_tree)

    
    # 评估函数
    def evaluate(self, X, y):
        count = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i])
            if y_pred == y[i]:
                count = count + 1
        print("predict accuracy: %f" % ((count * 1.0) / len(X)))


model = DecisionTreeClassifier(X[:-20], y[:-20])
model.train()
model.evaluate(X[:-20], y[:-20])