import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier



def buildData(filename, feature_cols, class_col):
	df = pd.read_csv(filename)
	feature_column_names = feature_cols
	predicted_class_name = class_col
	X = df[feature_column_names].values.astype(str)
	y = df[predicted_class_name].values.astype(str)
	return df, X, y

def splitTestTrain(df, X, y):
	split_test_size = 0.3
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = 42)
	print "{0:0.2f}% in training set".format((len(X_train)/float(len(df.index))) * 100), "\t{0:0.2f}% in test set".format((len(X_test)/float(len(df.index))) * 100)
	return X_train, X_test, y_train, y_test


def buildModel(clf, modelName, X_train, X_test, y_train, y_test):
	print "###",modelName,"###"
	clf = clf.fit(X_train.ravel(), y_train.ravel())
	predicted_train = clf.predict(X_train.ravel())
	print "Train Accuracy:\t",(np.mean(predicted_train == y_train.ravel()))*100, "%"
	predicted_test = clf.predict(X_test.ravel())
	print "Test Accuracy:\t",(np.mean(predicted_test == y_test.ravel()))*100, "%"

def drawLearningGraph():
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    
def getPrediction(filename, feature_cols, class_col):
	print "\n-----  Prediction on ", filename, "  ",feature_cols, " -----" 
	df, X, y = buildData(filename, feature_cols, class_col)
	X_train, X_test, y_train, y_test = splitTestTrain(df, X, y)
	text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
	buildModel(text_clf, "Multinomial Naive Bayes", X_train, X_test, y_train, y_test)
	text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
	buildModel(text_clf_svm, "SVM", X_train, X_test, y_train, y_test)

getPrediction("text_sentiment.csv", "content", "sentiment")
getPrediction("text_sentiment.csv", "content_modified", "sentiment")


# getPrediction("text_emotion_all.csv", "content", "sentiment")
# getPrediction("text_emotion.csv", "content_modified", "sentiment")
