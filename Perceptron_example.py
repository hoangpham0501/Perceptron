from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np 
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
# print(np.unique(y))	=> return the different class label stored in Iris ex: [0,1,2]
# target, we would see that the Iris ﬂower class names, Iris-Setosa, Iris-Versicolor,
#and Iris-Virginica, are already stored as integers (0, 1, 2), which is recommended
#for the optimal performance of many machine learning libraries.

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
# Using the train_test_split function from scikit-learn's cross_validation module
#we randomly split the X and y arrays into 30 percent test data (45 samples) and 70 percent training data (105 samples)

sc = StandardScaler()
# Using the preceding code, we loaded the StandardScaler class from the preprocessing module 
#and initialized a new StandardScaler object that we assigned to the variable sc. 

sc.fit(X_train)
# Using the fit method, StandardScaler estimated the parameters µ (sample mean) and σ (standard deviation) 
#for each feature dimension from the training data

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#By calling the transform method, we then standardized the training data using those estimated parameters µ and σ

ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
#after loading the Perceptron class from the linear_model module, we initialized a new Perceptron object 
#and trained the model via the fit method
# n_iter defines the number of epochs (passes over the training set)
# eta0 is equivalent to the learning rate eta that we used in our own perceptron implementation
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print("Misclassified samples: %d" %(y_test != y_pred).sum())
# we can make predictions via the predict method
# we see that the perceptron misclassifes 4 out of the 45 ﬂower samples.
# Thus, the misclassifcation error on the test dataset is 0.089 or 8.9% (4 / 45 ≈ 0.089).

print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
# we can calculate the classification accurancy of the perceptron on the test set.
# y_test are the true labels and y_pred are the class labels that we predicted

def plot_decision_region(X, y, classifier, test_idx = None, resolution = 0.02):
	#setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#plot the decision surface
	x1_min = X[:,0].min() - 1
	x1_max = X[:,0].max() + 1
	x2_min = X[:,1].min() - 1
	x2_max = X[:,1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#plot all samples
	X_test, y_test = X[test_idx, :], y[test_idx]
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

	#highlight test sample
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:,0], X_test[:,1], c = 'black', alpha = 1.0, linewidth = 1, marker = 'o', s = 55, label = 'test set')

# we can now specify the indices of the samples that we want to mark on the resulting plots
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X = X_combined_std, y = y_combined, classifier = ppn, test_idx = range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()