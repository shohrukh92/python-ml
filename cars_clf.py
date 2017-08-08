# -*- coding: utf-8 -*-
from sklearn import tree

def get_type(type_id):
    if type_id == 0:
        return "sports-car"
    elif type_id == 1:
        return "minivan"

# horsepowers, seats
features = [[300, 2], [450, 2], [200, 8], [150, 9]]
# 0 - sports-car, 1 - minivan
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

predicts = clf.predict([[400, 1], [600, 3], [100, 10], [230, 12]])

print(list(map(get_type, predicts)))
