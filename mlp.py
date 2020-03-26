from open import process
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


x_train, x_test, y_train, y_test = process()

# use one hot encoder for the labels
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)
integer_encoded = y_test.reshape(len(y_test), 1)
y_test = onehot_encoder.transform(integer_encoded)


def ch2_features(x_train, x_test, y_train):
    # select only important features using ch2
    ch2 = SelectKBest(chi2, k=1000)
    x_train = ch2.fit_transform(x_train, y_train)
    x_test = ch2.transform(x_test)
    x_train = x_train.toarray()
    return x_train, x_test


def svd_features(x_train, x_test):
    # dimensionality reduction using svd
    svd = TruncatedSVD(n_components=1000)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)
    return x_train, x_test


def pca_features(x_train, x_test):
    # dimensionality reduction using pca
    pca = PCA(n_components=1000)
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test


x_train, x_test = ch2_features(x_train, x_test, y_train)

# define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=1000))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(20, activation='softmax'))

# train the model and evaluate the result
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=15,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)

model.summary()