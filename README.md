# 自己动手从零统计学习代码实现

## 说明

### 第2章 感知机

```bash
user> python 2_Perceptron.py
predict accuracy: 0.950000
```
### 第3章 K近邻

```bash
user> python 3_KNN.py
# 取后20个作为测试样本
# model.evaluate(X[-20:], y[-20:])
predict accuracy: 0.950000
```

### 第4章 朴素贝叶斯

```bash
user> python 4_NaiveBayes.py
# 取后20个作为测试样本
# model.evaluate(X[-20:], y[-20:])
predict accuracy: 0.700000
```

### 第5章 决策树

```bash
user> python 5_DecisionTree.py
# 取后20个作为测试样本
# model.evaluate(X[-20:], y[-20:])
tree = {
	4: {
		'<= -0.19999481617148995': {
			1: {
				'<= -0.12218492411829279': {
					3: {
						'<= -1.4016953074469127': {
							2: {
								'<= -1.514340694423072': {
									0: {
										'<= -0.7465448320866089': 0,
										'> -0.7465448320866089': 0
									}
								},
								'> -1.514340694423072': 0
							}
						},
						'> -1.4016953074469127': 1
					}
				},
				'> -0.12218492411829279': 0
			}
		},
		'> -0.19999481617148995': 1
	}
}
predict accuracy: 0.892857
```



### 第5章 逻辑回归
```bash
user>python 6_LogistcRegression.py
0 : loss =  1119.531578297443
...
997 : loss =  1505.6267724606946
998 : loss =  1505.5914685824441
999 : loss =  1505.5565465625332
predict accuracy: 0.950000
```