# -*- coding: utf-8 -*-
import random
from scipy.spatial import distance
import numpy as np

def euc(a, b):
    return distance.euclidean(a, b)

print(euc([1,1], [0,0]))

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            #label = self.closest3(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        
        for i in range(1, len(self.X_train)):
            dist = euc(row, X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        
        return self.y_train[best_index]
    
    def closest3(self, row):
        d1 = d2 = d3 = euc(row, self.X_train[0])
        i1 = i2 = i3 = 0
        
        for i in range(1, len(self.X_train)):
            d = euc(row, X_train[i])
            if d < d1:
                d3 = d2
                i3 = i2
                d2 = d1
                i2 = i1
                d1 = d
                i1 = i
            elif d < d2:
                d3 = d2
                i3 = i2
                d2 = d
                i2 = i
            elif d < d3:
                d3 = d
                i1 = i
        
        labels = [self.y_train[i1], self.y_train[i2], self.y_train[i3]]
        arr = [0, 0, 0]
        arr[labels[0]] += 1
        arr[labels[1]] += 1
        arr[labels[2]] += 1
        
        return labels[np.argmax(arr)]
        

class ScrappyRandom():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions


from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .8)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
    
    
    
    
    
    
    
    
    
    
    
    