# -*- coding: utf-8 -*-
from sklearn import tree

# 0 - bumpy, 1 - smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 - apple, 1 - orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))