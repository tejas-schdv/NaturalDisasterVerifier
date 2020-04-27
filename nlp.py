import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import matplotlib.pyplot as plt
import string

from nltk.corpus import treebank
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def cleanUp(df_train, df_test):
    
    df = {}
    
    stopwords.words('english')
    
    # word_count
    df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
    df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))

    # unique_word_count
    df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
    df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

    # stop_word_count
    df_train['stop_word_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df_test['stop_word_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS])) 

    # url_count
    df_train['url_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df_test['url_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

    # mean_word_length
    df_train['mean_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df_test['mean_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # char_count
    df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
    df_test['char_count'] = df_test['text'].apply(lambda x: len(str(x)))
    
    df["df_train"] = df_train
    df["df_test"] = df_test
    
    return df


def main() :
    
    count_vectorizer = feature_extraction.text.CountVectorizer()
    clf = linear_model.RidgeClassifier()
    
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    train_label = train_df["target"]
    
    train_df["location"] = train_df["location"].fillna("no_location")
    test_df["location"] = test_df["location"].fillna("no_location")
    
    train_df["keyword"]  = train_df['keyword'].fillna("no_keyword")
    test_df["keyword"] = test_df['keyword'].fillna("no_keyword")
    
    dfHolder = cleanUp(train_df, test_df)
    
    train_df = dfHolder["df_train"]
    test_df = dfHolder["df_test"]
    
    print(train_df)
    
    
main()