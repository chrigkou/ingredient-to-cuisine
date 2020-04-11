# ingredient-to-cuisine

Cuisine categorization with neural networks using the [Recipe Ingredients Dataset](https://www.kaggle.com/kaggle/recipe-ingredients-dataset)!

# Description

- Step 1: Download the dataset from [here](https://www.kaggle.com/c/whats-cooking/data) 
- Step 2: Statisitcs are generated from the data such as nuber of cuisines and most popular cuisines
- Step 3: Text analysis
--punctuation removal, convert to lower case
--common stopword removal
--stemming using Porter Stemmer algorithm
--replace intergers with text
 - Step 4: Encode the cuisine column using Label Encoder from sklearn
 - Step 5: Split the data (X-ingredients after feature selection, Y-encoded labels)  into training and test sets (60:40)
- Step 6: Convert the train and test ingredients into TF-IDF matrices (fit for the training data in order to avoid creating correlation between the train and test)
- Step 7: Encode the labels using One Hot Encoder and perform dimensionality reduction using chi2 feature selection
- Step 8: Apply an MLP model using keras 
- Step 9: Separate two cuisines at a time from the dataset
- Step 10: Implement an SVM classifier with a linear kernel
- Step 11: Apply custom SVM and SVC classifier from sklearn to the data
- Step 12: Implement an RBF classifier using a gaussian and a square kernel

 
## File organization

 1. [open.py](https://github.com/chrigkou/ingredient-to-cuisine/blob/master/open.py): data statistics, text analysis and preprocessing, split in train and test sets, use label encoder for the cuisines
 2. [mlp.py](https://github.com/chrigkou/ingredient-to-cuisine/blob/master/mlp.py): dimensionaltity reduction with chi2 feature selection, svd decomposition and pca. Apply a 4 layer MLP on the data
 3.  [svm.py](https://github.com/chrigkou/ingredient-to-cuisine/blob/master/svm.py): feature selection with chi2, select two classes from the dataset for 1 vs 1 SVM classifier. The custom SVM is a mathematic implementation using matrices. The results are compared to the SVC classifier from sklearn.
 4. [rbf.py](https://github.com/chrigkou/ingredient-to-cuisine/blob/master/rbf.py): the RBF network is implemented with a gaussian and a square kernel. The initial centers are selected randomly or using kmeans.

 

## Run an experiment

In order to train and test a selected network all you have to do is run the corrisponding python file.

> **Note:** You can change the method of feature selection for each experimenrt to test different algorithms and networks.


## MLP topology

The topology of the MLP network 

Model: sequential
|           Layer(type)    |Output Shape                          |Param #                        |
|----------------|-------------------------------|-----------------------------|
|dense_78 (Dense) |(None, 64)            |64064            |
|dropout_37(Dropout)         |(None, 64)             |0          |
|dense_79(Dense)          |(None, 32)|2080|
|dense_80 (Dense) |(None, 20)            |660|

