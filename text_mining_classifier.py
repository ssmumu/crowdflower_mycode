import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import re
from numpy import array
from sklearn.linear_model import SGDClassifier

df = pd.read_csv("text_sentiment_copywhole.csv", low_memory=False)
feature_column_names = ['content']
# feature_column_names = ['content_modified']
predicted_class_name = ['sentiment']
X = df[feature_column_names].values.astype(str)
y = df[predicted_class_name].values.astype(str)


split_test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = 42)
print "{0:0.2f}% in training set".format((len(X_train)/float(len(df.index))) * 100)
print "{0:0.2f}% in test set".format((len(X_test)/float(len(df.index))) * 100)


def buildModel(clf, modelName):
	print "\n###",modelName,"###"
	clf = clf.fit(X_train.ravel(), y_train.ravel())
	predicted_train = clf.predict(X_train.ravel())
	print "Train Accuracy: ",(np.mean(predicted_train == y_train.ravel()))*100, "%"
	predicted_test = clf.predict(X_test.ravel())
	print "Test Accuracy: ",(np.mean(predicted_test == y_test.ravel()))*100, "%"


text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
buildModel(text_clf, "Multinomial Naive Bayes")

text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
buildModel(text_clf_svm, "SVM")
