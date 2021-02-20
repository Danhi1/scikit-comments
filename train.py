import numpy as np
import pickle 
from builddata import Review
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score

# Data loadpath
LOADPATH = "./data/reviews.npy"
# Classifier savepath
CLF_SAVEPATH = "./Models/scikit-comments.pkl"


print("Loading data...")
reviews = np.load(LOADPATH, allow_pickle = True)

print("Splitting data...")
training, testing = train_test_split(reviews, test_size = 0.25, random_state = 10)

train_x = [x.text for x in training]
train_y = [x.opinion for x in training]

test_x = [x.text for x in testing]
test_y = [x.opinion for x in testing]

print("Vectorizing data...")
vectorizer = TfidfVectorizer()

train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

print("Fitting the classifier...")
classifier = svm.SVC(kernel = 'rbf')
classifier.fit(train_x_vectors, train_y)

print("Checking the metrics...")
print("Accuracy: ", classifier.score(test_x_vectors, test_y))
print("F1 Score: ", f1_score(test_y, classifier.predict(test_x_vectors), average = None))

if CLF_SAVEPATH:
    with open(CLF_SAVEPATH, "wb") as f:
        pickle.dump(classifier, f)
        pickle.dump(vectorizer, f)
        print("Model saved as:" , CLF_SAVEPATH)
