import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

#print(train_df.shape)
#
#print(train_df[train_df["target"] == 0]["text"].values[1])
#print(train_df[train_df["target"] == 1]["text"].values[1])
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

#print(example_train_vectors[0].todense().shape)
#print(example_train_vectors[0].todense())

train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

print(scores)

clf.fit(train_vectors, train_df["target"])

print(clf.predict(test_vectors))