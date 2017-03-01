import pandas as pd
import csv
from string import ascii_letters
from collections import Counter, defaultdict
import re, hashlib, os
from pandas.tseries.offsets import *
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from scipy.sparse import hstack
import numpy as np
import math
import sys


def stats(feats):
    """ Dataset stats """

    print("Number of posts with Latin: {0} / {1}".format(sum(feats['latin_cs']), len(feats)))
    #print("Number of posts with English: {0} / {1}".format(sum(feats['en_cs']), len(feats)))

    # Number of threads
    print("#threads: {:d}".format(len(set(zip(feats['article'], feats['thread_title'])))))

    # Number of editors
    print('#editors: {:d}'.format(len(set(feats['editor']))))
    print()

""" Correlation and t-test """
def evaluate(feats):

    print("STATISTICAL ANALYSIS:")

    binary_feats = ['latin_cs', 'other_latin_cs', 'editor_two_quotes',
                    'other_two_quotes']
    continuous_feats = ['#editor_turns', '#other_turns', 
                        'editor_prop_latin', 'other_prop_latin',
                        'editor_prop_switches', 'other_prop_switches',]
                        #'editor_en_technical', 'other_en_technical']

    # ## Variation among cs features
    # Discrete features--t test
    for f in binary_feats:
        print(f)
        fpresent = feats[feats[f]==True]
        nof = feats[feats[f]==False]
        print("Number of positive feature example: {:d}".format(len(fpresent)))
        print("Positive feature mean score: {:f}".format(np.mean(fpresent['editor_score'])))
        print("Negative feature mean score: {:f}".format(np.mean(nof['editor_score'])))
        print(ttest_ind(fpresent['editor_score'], nof['editor_score']))
        print()

    # Continuous features (measure correlation)
    for f in continuous_feats:
        print(f)
        print('#nonzero: {:d}'.format(np.nonzero(feats[f])[0].shape[0]))
        r, p = pearsonr(feats[f], feats['editor_score'])
        print("r={0}, p={1}".format(r,p))
        print()


""" Prediction """
def prediction(feats):

    print("CLASSIFICATION/REGRESSION:")

    # Select columns
    ed_nonbow_cols = ['latin_cs', 'editor_prop_latin', 'editor_prop_switches', 
            'editor_two_quotes']
    other_nonbow_cols = ['other_latin_cs', 'other_prop_latin', 
            'other_prop_switches', 'other_two_quotes']

    # Vectorize input features
    feats_v = {}

    # Get unigram features
    #v = TfidfVectorizer(min_df=1, stop_words='english') 
    v = TfidfVectorizer(min_df=1) 
    #v = CountVectorizer(min_df=1) 
    # don't think has Arabic stopwords
    edbow = v.fit_transform(feats['editor_talk'])

    #v_other = TfidfVectorizer(min_df=1, stop_words='english')
    v_other = TfidfVectorizer(min_df=1)
    #v_other = CountVectorizer(min_df=1)
    other_bow = v_other.fit_transform(feats['other_talk'])

    # Get exclusive editor non-unigram features
    ed_nonbow_d = {}
    for col in ed_nonbow_cols:
        ed_nonbow_d[col] = np.array([feats[col]]).T
    ed_nonbow = np.hstack(ed_nonbow_d.values())

    # Get others' non-unigram features
    nonbow_d = {}
    for col in other_nonbow_cols:
        nonbow_d[col] = np.array([feats[col]]).T
    #     nonbow_d[col] = np.array([(v - min(feats[col]))/max(feats[col]) for v in feats[col].values]).T
    nonbow = np.hstack(nonbow_d.values())
    #print(nonbow.shape)

    # Assemble editor features
    edfeats = hstack([edbow, ed_nonbow])

    # Assemble non-unigram features
    nonbow_f = np.hstack([ed_nonbow, nonbow])

    # Assemble unigram features
    bow_f = hstack([edbow, other_bow])
    #print(bow_f.shape)

    # Assemble all features
    feats_v = hstack([edbow, other_bow, ed_nonbow, nonbow])
    feats_v.shape

    # Train and test classifiers
    train_test(feats, edbow, 'editor unigrams')
    train_test(feats, ed_nonbow, 'editor CS')
    train_test(feats, edfeats, 'editor unigrams+CS')
    train_test(feats, bow_f, 'all unigrams')
    train_test(feats, nonbow_f, 'all CS')
    train_test(feats, feats_v, 'all unigrams+CS')


def train_test(feats, train_data, desc):

    # Baseline for classification
    #baseline(feats)

    # Train and test logistic regression classifier
    ##clf = LogisticRegression(class_weight='balanced')
    #clf = LogisticRegression()
    ##loo = LeaveOneOut().split(train_data)
    #scores = cross_val_score(clf, train_data, feats['editor_success'], cv=10)
    #print("%s Accuracy: %0.3f" % (desc, scores.mean()))

    # Train and test linear regression classifier
    clf = LinearRegression()
    pred = cross_val_predict(clf, train_data, feats['editor_score'], cv=10)
    scores = mean_squared_error(feats['editor_score'], pred)
    print("%s MSE: %0.3f" % (desc, scores.mean()))


def baseline(feats):
    # right now just for classification
    print("Majority class guess {:0.3f} accuracy".format(
            max(1-sum(feats['editor_success'])/len(feats),
                sum(feats['editor_success']/len(feats)))))


def main():
    featfile = sys.argv[1] # input file
    
    # Load features
    feats = pd.read_csv(featfile)

    stats(feats)
    evaluate(feats)
    prediction(feats)

if __name__ == '__main__':
    main()
