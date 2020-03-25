import json
import string
from nltk.stem.porter import *
import inflect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
import collections


# this function is used for getting some statistics from the data
def data_stats():
    with open('train.json', encoding='utf-8') as fp:
        dict_data = json.load(fp)

    print("Data example ")
    print(dict_data[0])
    print("Data length ", len(dict_data))

    # create separate lists containing the data
    cuisine = []
    ingredients = []
    for i in range(len(dict_data)):
        cuisine.append(dict_data[i]['cuisine'])
        ingredients.append(dict_data[i]['ingredients'])

    cuisines_count = collections.Counter(cuisine)
    print("Number of different Cuisines ", len(cuisines_count.values()))
    common = cuisines_count.most_common(5)
    print("The 5 most popular cuisines  ", common)


# read and process the data before using them for an algorithm
def process():
    # The dataset was chosen from kaggle.com - Recipe Ingredients Dataset
    with open('train.json', encoding='utf-8') as fp:
        dict_data = json.load(fp)

    with open('test.json', encoding='utf-8') as fp1:
        dict_test = json.load(fp1)

    # create separate lists containing the data
    cuisine = []
    ingredients = []
    for i in range(len(dict_data)):
        cuisine.append(dict_data[i]['cuisine'])
        ingredients.append(dict_data[i]['ingredients'])

    # get a list containing all the recipes
    ingred = []
    for s in ingredients:
        # create one string for the ingredients of each recipe
        s = ' '.join(s)
        ingred.append(s)

    ingredients.clear()

    # Data pre processing

    # list of common stopwords
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    recipes = []
    for recipe in ingred:
        # Remove punctuation and set to lowercase
        punctuation = set(string.punctuation)
        doc = ''.join([w for w in recipe.lower() if w not in punctuation])

        # Stopword removal
        doc = [w for w in doc.split() if w not in stopwords]

        # Stemming
        stemmer = PorterStemmer()
        doc = [stemmer.stem(w) for w in doc]

        # replace integers with text
        en = inflect.engine()
        doc = [w if not w.isdigit() else en.number_to_words(w) for w in doc]

        # Convert list of words to one string
        doc = ' '.join(w for w in doc)

        recipes.append(doc)

    # encode every label with a value between 0 and n_classes-1(there 20 different classes in total)
    encoder = preprocessing.LabelEncoder()
    cuisine = encoder.fit_transform(cuisine)

    # split in train and test sets
    x_train, x_test, y_train, y_test = train_test_split(recipes, cuisine, test_size=0.4, random_state=100)

    # encode every recipe - convert a collection of recipes to a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    return x_train, x_test, y_train, y_test


data_stats()
x_train, x_test, y_train, y_test = process()

