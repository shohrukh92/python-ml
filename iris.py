# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])

for i in range(len(iris.target)):
    pass
    # print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
    
# delete 0, 50 and 100th items in corresponding arrays 
# to split data for training and prediction
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx] # get items in 0, 50, 100 index position
test_data = iris.data[test_idx] # get items in 0, 50, 100 index position

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# viz code
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")  



