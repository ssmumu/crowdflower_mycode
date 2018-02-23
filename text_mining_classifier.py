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

df = pd.read_csv("text_emotion.csv")
feature_column_names = ['content_modified']
predicted_class_name = ['sentiment']
X = df[feature_column_names].values
y = df[predicted_class_name].values

# regex = re.compile('[^a-zA-Z0-9 ]')
# X = [[regex.sub('',(''.join(str(x))))] for x in X]
# X = array(X)

split_test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = 42)
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train.ravel())
# print(X_train_counts.shape)

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)


# clf = MultinomialNB().fit(X_train_tfidf, y_train.ravel())



text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
text_clf = text_clf.fit(X_train.ravel(), y_train.ravel())
predicted = text_clf.predict(X_test.ravel())
print( "MultinomialNB accuracy: ",(np.mean(predicted == y_test.ravel()))*100, "%")

nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())
prediction_from_test_data = nb_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test.ravel(), prediction_from_test_data)
print("GaussianNB accuracy: ",accuracy*100,"%")

