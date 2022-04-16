from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from LogisticRegressionGD import LogisticRegressionGD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
        

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
#print (X)
y= iris.target
#print (y)
#print ('Class labels:', np.unique(y))
#np.unique returns the unique values stored in y. They are the 3 different flower names 
#but stored as integers (0,1,2)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1, stratify=y)

# test size 0.3 means that 30% of the data will be used as test - in this case we have 150 samples
# 105 will be used for training and 45 for testing
# Also the train_test_split shuffles the data before splitting them
# random_state=1 provides a fixed random seed for the internal randomizing algorithm 
# stratify = y means that the split support stratification - Stratification means that the
# train_test_split method returns training and test subsets that have the same proportions of 
# class labels as the input dataset - eg. 50% Setosas, 50% Virginica


#print ('Labels counts in y:', np.bincount(y))
#print ('Labels count in y_test:', np.bincount(y_test))
#print ('Labels count in X_test:', np.bincount(y_train))

sc=StandardScaler()
sc.fit(X_train)
X_train_std= sc.transform(X_train)
X_test_std= sc.transform(X_test)

#The fit method estimates the mean value and standart deviation -> Will be used for the standartization
#Transform -> Standartize the data using the μ and σ we got from the previous step 
#Note: we use the same μ and σ for our data because we need them to be comparable 

ppn = Perceptron(eta0=0.1, max_iter=1000, random_state=1)
ppn.fit(X_train_std,y_train)

# We create a perceptron object -> we give an eta and a random_state and max_iter and we call the fit function
# Now we have trained our model with the above code - we can now make predictions

y_pred=ppn.predict(X_test_std)
print ('Missclassified Examples: %d' % (y_test != y_pred).sum())

# we do the prediction and we check the differences of those predictions with our test y data

print ('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

# we use the metric libry to check the percendage accuracy
# Below I will create the graphical represendation of the results by using a plot_decision_regions


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

#plot_decision_regions(X=X_combined_std, y=y_combined,
                  #    classifier=ppn, test_idx=range(105, 150))
#plt.xlabel('petal length [standardized]')
#plt.ylabel('petal width [standardized]')
#plt.legend(loc='upper left')

#plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
#plt.show()

#Implementation of the LogisticRegression method is below - the class has been imported
X_train_01_subset=X_train[(y_train==0)|(y_train==1)]
y_train_01_subset=y_train[(y_train==0)|(y_train==1)]

# Above, its creating 2 new lists with only the 2 of the 3 classes y==0 or y==1
# The logistic regression model works only for binary tasks hence why the above step needs to be taken

"""
lrgd=LogisticRegressionGD(eta=0.05,
                          n_iter=1000,
                          random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)
plt.xlabel('petal length [standartized]')
plt.ylabel('petal width [standartized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""
# LogisticRegression build in in the code below
# importing sklearn.lenear_model

lr= LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

#C is not the iterations
lr.fit(X_train_std,y_train)

print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))
print(lr.predict(X_test_std[:3, :]))

#something else that might be needed is to convert a single row array entry into a two dimentional data array by using Numpy
print(lr.predict(X_test_std[0,:].reshape(1,-1)))

"""
plot_decision_regions(X_combined_std,
       y_combined,
       classifier=lr,
       test_idx=range(105,150))
"""

# the reason why I am parsing the combined array is because I am setting the test range in the variables as well
"""
plt.xlabel('petal length [standartized]')
plt.ylabel['petal width [standartized]']
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""

#regularization L2
# To C=10.**c -> gives different power values to C for it to contantly change. eg. 10^0, 10^1 etc
# Also in this case based on the for condition we go from 10^-4 to 10^4

"""
weights, params =[], []
for c in np.arange(-5,5):
    lr=LogisticRegression(C=10.**c, random_state=1, solver='lbfgs',multi_class='ovr')
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1]) #adding the coefficient in the array for later use in the graph,
                                # I am also getting only one class coeficient 
    print(lr.coef_[1], 10.**c)
    params.append(10.**c)   #we add the parameter in an array for later use in the graph
weights=np.array(weights)

plt.plot(params,weights[:,0],
         label='petal length')
plt.plot(params,weights[:,1],
         linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
"""
#implementation of SVM - used to minimize the affect non-linear data would have to our weights
# it can be used via the SVC class below

"""
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
plt.show()

# For very large data sets that do not fit in the pc memory we can use the below 
# class which is [art of SGDClassifier 

ppn= SGDClassifier(loss = 'perceptron')
lr=SGDClassifier(loss='log')
svm=SGDClassifier(loss='hinge')

"""
#randomised XOR values
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
plt.show()

#instead of kernel liner we have kernel 'rbf' now
svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_xor.png', dpi=300)
plt.show()


