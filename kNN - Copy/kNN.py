# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:56:42 2017

@author: don
"""
import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import collections as counter
os.chdir('C:\\Users\don\Desktop\kNN classifiers\Giant Table\kNN - Copy')


names=['Direction/Sensor/Bill', 'Detection', 'tach1', 'tach2', 'tach3', 'tach4', 
'tach5', 'tach6', 'tach7', 'tach8', 'tach9', 'tach10', 'tach11', 'tach12', 
'tach13', 'tach14', 'tach15', 'tach16', 'tach17', 'tach18', 'tach19', 
'tach20', 'tach21', 'tach22', 'tach23', 'tach24', 'tach25', 'tach26', 
'tach27', 'tach28', 'tach29', 'tach30', 'tach31', 'tach32', 'tach33', 
'tach34', 'tach35', 'tach36', 'tach37', 'tach38', 'tach39', 'tach40', 
'tach41', 'tach42', 'tach43', 'tach44', 'tach45', 'tach46', 'tach47', 
'tach48', 'tach49', 'tach50', 'tach51', 'tach52', 'tach53', 'tach54', 
'tach55', 'tach56', 'tach57', 'tach58', 'tach59', 'tach60', 'tach61', 
'tach62', 'tach63', 'tach64', 'tach65', 'tach66', 'tach67', 'tach68', 
'tach69', 'tach70', 'tach71', 'tach72', 'tach73', 'tach74', 'tach75', 
'tach76', 'tach77', 'tach78', 'tach79', 'tach80', 'tach81', 'tach82', 
'tach83', 'tach84', 'tach85', 'tach86', 'tach87', 'tach88', 'tach89', 
'tach90']
df = pd.read_csv('giant_table_nohead_DSB_numbered_RD.csv', header=None, names=names)



"""Now, it’s time to get our hands wet. We’ll be using scikit-learn 
to train a KNN classifier and evaluate its performance on the data set 
using the 4 step modeling pattern:

Import the learning algorithm
Instantiate the model
Learn the model
Predict the response
scikit-learn requires that the design matrix XX and 
target vector yy be numpy arrays so let’s oblige. Furthermore, we need 
to split our data into training and test sets. 
The following code does just that."""

# create design matrix X and target vector y
X = np.array(df.ix[:, 2:92]) 	# end index is exclusive (Tachs 1-90)
y = np.array(df['Detection']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=42)



"""
Finally, following the above modeling pattern, we define our classifer, 
in this case KNN, fit it to our training data and evaluate its accuracy. 
We’ll be using an arbitrary K but we will see later on how cross validation 
can be used to find its optimal value."""
# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print("Accuracy: " + str(accuracy_score(y_test, pred)))



"""
Cross-Validation and Parameter Fitting
"""

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())




"""
Finally: PLOT Misclassification error vs k


# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

"""

def train(X_train, y_train):
	# do nothing 
	return


def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return counter.Counter(targets).most_common(1)[0][0]



def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# train on the input data
	train(X_train, y_train)

	# loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))





predictions = []

kNearestNeighbor(X_train, y_train, X_test, predictions, 7)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is %d%%' % accuracy*100)





