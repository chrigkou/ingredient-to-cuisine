from sklearn.svm import SVC
from open import process
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix


def ch2_features(x_train, x_test, y_train):
    # select only important features using ch2
    ch2 = SelectKBest(chi2, k=1000)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test = ch2.transform(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    return x_train, x_test


# get the data from 2 selected classes from the train and test data
def extract_classes(x_train, x_test, y_train, y_test, id_1, id_2):
    new_train = []
    new_label = []
    new_test = []
    new_test_label = []
    i = 0
    for label in y_train:
        if label == id_1:
            new_train.append(x_train[i])
            new_label.append(-1)
        elif label == id_2:
            new_train.append(x_train[i])
            new_label.append(1)
        i = i+1

    k = 0
    for label in y_test:
        if label == id_1:
            new_test.append(x_test[k])
            new_test_label.append(-1)
        elif label == id_2:
            new_test.append(x_test[k])
            new_test_label.append(1)
        k = k + 1
    return new_train, new_test, new_label, new_test_label


# fit the data for 1 vs 1 linear svm classifier
def svm_fit(x_train, y_train):

    x_train = np.stack(x_train, axis=0)
    y_train = np.stack(y_train, axis=0)
    n_samples, n_features = np.shape(x_train)
    kernel = np.zeros((n_samples, n_samples))

    # the (linear) kernel matrix
    for i in range(n_samples):
        for j in range(n_samples):
            kernel[i, j] = np.dot(x_train[i], x_train[j])

    P = matrix(np.outer(y_train, y_train) * kernel)
    c = matrix(-np.ones(n_samples))
    # add a small value to avoid problem when Hessian is badly conditioned
    P = P + 1e-10 * matrix(np.identity(n_samples))
    h = matrix(np.zeros(n_samples))
    # this matrix type needs to be double in order to find the optimal solution
    A = matrix(y_train, (1, n_samples), tc='d')
    b = matrix(0.0)
    G = matrix(np.identity(n_samples) * -1)
    # reduce the number of max iterations
    solvers.options['maxiters'] = 50
    solution = solvers.qp(P, c, G, h, A, b)

    # get the Lagrange multipliers from the solution of the optimization problem
    a = np.ravel(solution['x'])
    # keep the indexes of non-zero lagrange multipliers
    idx = a > 1e-8
    # get the non zero lagrange multipliers
    lagrange_mult = a[idx]
    # get the support vectors and their labels
    supp_vectors = x_train[idx]
    supp_labels = y_train[idx]

    # calculate the weights
    # using the equation w = Σak*yk*xk where k=1 to number of features
    w = np.zeros(n_features)
    print("Number of support vectors", len(lagrange_mult))
    for k in range(len(lagrange_mult)):
        w += lagrange_mult[k] * supp_labels[k] * supp_vectors[k]

    # calculate the bias
    b = 0
    bound = min(n_features, len(lagrange_mult))
    for k in range(bound):
        b += supp_labels[k]
        b -= np.sum(supp_vectors[k] * w[k])
    # a regularized bias seems to yield better results
    b = b / bound

    return w, b


# predict the labels of  new samples
def svm_predict(x_test, w, b):
    # use the equation y= w*x + b where y=1 or y=-1
    classif = np.sign(np.dot(x_test, w)+b)
    return classif


x_train, x_test, y_train, y_test = process()
x_train, x_test = ch2_features(x_train, x_test, y_train)
x_train, x_test, y_train, y_test = extract_classes(x_train, x_test, y_train, y_test, 0, 1)

# using custom svm
w, b = svm_fit(x_train, y_train)
y_pred = svm_predict(x_test, w, b)
print("Using custom svm: ")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# using sklearn svm classifier
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
print("Using svm from sklearn library: ")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
