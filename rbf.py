from sklearn.cluster import KMeans
import numpy as np
from open import process
from sklearn.feature_selection import SelectKBest, chi2
from scipy.linalg import norm, pinv
import scipy
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import OneHotEncoder


def ch2_features(x_train, x_test, y_train):
    # select only important features using ch2
    ch2 = SelectKBest(chi2, k=1000)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test = ch2.transform(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    return x_train, x_test


# use random samples from the train set as the centers
def rand_centers(x_train, clusters_num):
    xrange = len(x_train)
    centr_ids = random.sample(range(xrange), clusters_num)
    centers = [x_train[i, :] for i in centr_ids]
    return centers


# use kmeans to compute the centers
def kmeans_centers(x_train, y_train, clusters_num):
    kmeans = KMeans(n_clusters=clusters_num, max_iter= 100)
    kmeans.fit(x_train, y_train)
    centers = kmeans.cluster_centers_
    return centers


# the gaussian kernel for the rbf
def gaussian_kernel(center, sample):
    kern = np.exp(- norm((center - sample) ** 2, 2))  # using Euclidean distance
    return kern


# kernel using the square function for the rbf
def square_kernel(center, sample):
    kern = norm((center - sample) ** 2, 2)  # using Euclidean distance
    return kern


def rbf_fit(x_train, y_train, centers):

    # keep a matrix for the distance of each sample from each center using the kernel
    F = np.zeros((x_train.shape[0], len(centers)), float)
    for ci, c in enumerate(centers):
        for xi, x in enumerate(x_train):
            F[xi, ci] = gaussian_kernel(c, x)
    W = scipy.dot(pinv(F), y_train)  # calculate the weights
    return W


def rbf_predict(x_test, W, centers):
    # keep a matrix for the distance of each sample from each center
    F = np.zeros((x_test.shape[0], len(centers)), float)
    for ci, c in enumerate(centers):
        for xi, x in enumerate(x_test):
            F[xi, ci] = gaussian_kernel(c, x)
    y_pred = scipy.dot(F, W)
    return y_pred


x_train, x_test, y_train, y_test = process()
x_train, x_test = ch2_features(x_train, x_test, y_train)  # reduce the dimensions of the data

# use one hot encoder for the labels
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)
integer_encoded = y_test.reshape(len(y_test), 1)
y_test = onehot_encoder.transform(integer_encoded)

centers = kmeans_centers(x_train, y_train, 200)
weights = rbf_fit(x_train, y_train, centers)
y_pred = rbf_predict(x_test, weights, centers)
y_pre = np.argmax(y_pred, axis=1)  # transform the array by choosing the most probable class for each sample
print("Rbf with kmeans centers: ")
print(accuracy_score(integer_encoded, y_pre)*100)

centers = rand_centers(x_train, 200)
weights = rbf_fit(x_train, y_train, centers)
y_pred = rbf_predict(x_test, weights, centers)
y_pre = np.argmax(y_pred, axis=1)
print("Rbf with random centers: ")
print(accuracy_score(integer_encoded, y_pre)*100)

