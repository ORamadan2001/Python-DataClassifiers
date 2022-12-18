#!/usr/bin/env python
# coding: utf-8

# Import statements
import numpy as np
import pandas as pd
import time
import random

from scipy.sparse import csr_matrix
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier


# Options 
trainfilename = "train.txt"
testfilename = "test.txt"
printCV = True
sample = 'Oversample'


# Import training data (presented in sparse matrix with class label as first index)
raw_df = pd.read_csv(trainfilename, header = None)
#raw_df.head()

classes = []
for row in raw_df[0]:
    rowlist = row.split()
    classes.append(int(rowlist[0]))

raw_df['class'] = classes

maxClassSize = raw_df['class'].value_counts().max()
minClassSize = raw_df['class'].value_counts().min()
print('Max class size: ', maxClassSize)
print('Max class size: ', minClassSize)

# Oversample data to balance classes
if sample == 'Oversample':
    print('Oversampling...')
    lst = [raw_df]
    for c, group in raw_df.groupby('class'):
        lst.append(group.sample(maxClassSize - len(group), replace = True))
    df = pd.concat(lst)
elif sample == 'Undersample':
    print('Undersampling...')
    trueSamples = raw_df[raw_df['class'] == 1]
    falseSamples = raw_df[raw_df['class'] == 0]
    ts = trueSamples.sample(n = minClassSize, random_state = 99)
    fs = falseSamples.sample(n = minClassSize, random_state = 99)
    df = pd.concat([ts, fs], axis = 0)
else:
    print('Sample type not valid (', sample, '): Using raw data.')
    df = raw_df

df.head()

# Import test data
test_df = pd.read_csv(testfilename, header = None)
#df.head()

# Parse training data into sparse matrix format
classes = []
indices_list = []
for row in df[0]:
    rowlist = row.split()
    classes.append(int(rowlist[0]))
    indices = rowlist[1:]
    indices_list.append(indices)

# Get parameters for sparse matrix size
MAXWIDTH = 0
for indices in indices_list:
    for index in indices:
        if int(index) > MAXWIDTH:
            MAXWIDTH = int(index)
MAXHEIGHT = len(df[0])
print('WIDTH HEIGHT:', MAXWIDTH, MAXHEIGHT)

row  = []
col  = []
data = []

for i in range(len(indices_list)):
    for j in range(len(indices_list[i])):
        row.append(i)
        col.append(int(indices_list[i][j]))
        data.append(1)
    
if len(row) != len(col) and len(col) != len(data):
    print('Error: mismatched length of instantiation parameters lists')

# Create sparse matrix of MAXWIDTH, MAXHEIGHT and datatype int32
train_matrix = csr_matrix((data, (row, col)))


# Parse test data into sparse matrix format
test_indices_list = []
for row in test_df[0]:
    rowlist = row.split()
    test_indices_list.append(rowlist)
    
MAXWIDTH = 0
for indices in test_indices_list:
    for index in indices:
        if int(index) > MAXWIDTH:
            MAXWIDTH = int(index)
MAXHEIGHT = len(test_df[0])
print('WIDTH HEIGHT:', MAXWIDTH, MAXHEIGHT)

row  = []
col  = []
data = []

for i in range(len(test_indices_list)):
    for j in range(len(test_indices_list[i])):
        row.append(i)
        col.append(int(test_indices_list[i][j]))
        data.append(1)
    
if len(row) != len(col) and len(col) != len(data):
    print('Error: mismatched length of instantiation parameters lists')

# Create sparse matrix of MAXWIDTH, MAXHEIGHT and datatype int32
test_matrix = csr_matrix((data, (row, col)))

# Create a decision tree classifier using sklearn
tc = tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'best', random_state = 1)
tc = tc.fit(train_matrix, classes)
#prediction = tc.predict(train_matrix.getrow(i))

# Create a neural network classifier
nc = MLPClassifier(hidden_layer_sizes = 100, max_iter = 100, random_state = 1)
nc = nc.fit(train_matrix, classes)

# Define a function for easy cross validation
def cv(model, trainingData, trainingClasses, folds = 3, es = np.nan):
    results = cross_validate(model, trainingData, trainingClasses, 
                          cv = folds, scoring = ['accuracy', 'precision', 'recall', 'f1'], return_train_score = True, error_score = es)
    if printCV:
        print("Test accuracy: ", results['test_accuracy'].mean())
        print("Test precision: ", results['test_precision'].mean())
        print("Test recall: ", results['test_recall'].mean())
        print("Test F1: ", results['test_f1'].mean())
    return results

print('Decision Tree Classifier: ')
cv(tc, train_matrix, classes, folds = 10)

print('\nNeural Network Classifier: ')
cv(nc, train_matrix.toarray(), classes, folds = 10, es = 'raise')
print()

# Create a decision tree classifier using sklearn
tc = tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', random_state = 1)
tc = tc.fit(train_matrix, classes)
#prediction = tc.predict(train_matrix.getrow(i))

# Create a neural network classifier
nc = MLPClassifier(hidden_layer_sizes = 50, max_iter = 100, random_state = 1)
nc = nc.fit(train_matrix, classes)

print('Decision Tree Classifier: ')
cv(tc, train_matrix, classes, folds = 10)

print('\nNeural Network Classifier: ')
cv(nc, train_matrix.toarray(), classes, folds = 10, es = 'raise')
print()

# Create a decision tree classifier using sklearn
tc = tree.DecisionTreeClassifier(criterion = 'log_loss', splitter = 'best', random_state = 1)
tc = tc.fit(train_matrix, classes)
#prediction = tc.predict(train_matrix.getrow(i))

# Create a neural network classifier
nc = MLPClassifier(hidden_layer_sizes = 100, max_iter = 200, random_state = 1)
nc = nc.fit(train_matrix, classes)



print('Decision Tree Classifier: ')
cv(tc, train_matrix, classes, folds = 10)

print('\nNeural Network Classifier: ')
cv(nc, train_matrix.toarray(), classes, folds = 10, es = 'raise')
print()

f = open('output_hw3.txt', 'w')
startTime = time.time()
for i in range(len(test_df[0])):
    prediction = str(tc.predict(test_matrix.getrow(i)))[1]
    print(prediction, end = ', ')
    f.write(str(prediction) + '\n')
endTime = time.time()
f.close()
minutes = (endTime - startTime)/60
print('\nElapsed time (mins): ' + str(minutes))