import pandas as pd
import csv
from string import ascii_letters
from nltk.corpus import words, stopwords
import nltk
from collections import Counter, defaultdict
import re, hashlib, os
endict = set(words.words())
from IPython.core.debugger import Tracer; debug_here = Tracer()
from pandas.tseries.offsets import *
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn import cross_validation, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, LeaveOneOut
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from scipy.sparse import hstack
import numpy as np
import math
import sys

""" I/O definitions """
#featfile = 'cs_discussion_features+2000.csv'
featfile = sys.argv[1]

# Load features
feats = pd.read_csv(featfile)

""" Dataset stats """
def stats():

    print("Number of posts with English: {0} / {1}".format(sum(feats['en_cs']), len(feats)))

    # Number of threads
    print("#threads: {:d}".format(len(set(zip(feats['article'], feats['thread_title'])))))

    # Number of editors
    print('#editors: {:d}'.format(len(set(feats['editor']))))

""" Correlation and t-test """
def eval():

    binary_feats = ['en_cs', 'other_en_cs', 'editor_two_quotes',
                    'other_two_quotes']
    continuous_feats = ['editor_prop_en', 'other_prop_en',
                        'editor_prop_switches', 'other_prop_switches',
                        'editor_en_technical', 'other_en_technical']

    # ## Variation among cs features
    # Discrete features--t test
    for f in binary_feats:
        print(f)
        fpresent = feats[feats[f]==True]
        nof = feats[feats[f]==False]
        print("Number of positive feature example: {:d}".format(len(fpresent)))
        print(ttest_ind(fpresent['editor_score'], nof['editor_score']))
        print()

    # Continuous features (measure correlation)
    for f in continuous_feats:
        print(f)
        print('#nonzero: {:d}'.format(np.nonzero(feats[f])[0].shape[0]))
        print(pearsonr(feats[f], feats['editor_score']))
        print()


""" Prediction """

def prediction():
    # Select columns
    #ed_nonbow_cols = ['#editor_turns', 'en_cs', 'editor_prop_en', 'editor_prop_switches', 'editor_two_quotes', 
                      #'editor_en_technical']
#    other_nonbow_cols = ['#other_turns', 'other_en_cs', 'other_prop_en', 'other_prop_switches', 'other_two_quotes', 
#                      'other_en_technical']
    ed_nonbow_cols = ['en_cs', 'editor_prop_en', 'editor_prop_switches', 'editor_two_quotes', 
                      'editor_en_technical']
    other_nonbow_cols = ['other_en_cs', 'other_prop_en', 'other_prop_switches', 'other_two_quotes', 
                      'other_en_technical']

    # Vectorize input features
    feats_v = {}

    # Get unigram features
    v = TfidfVectorizer(min_df=1, stop_words='english')
    edbow = v.fit_transform(feats['editor_talk'])
    #print(edbow.shape)

    v_other = TfidfVectorizer(min_df=1, stop_words='english')
    other_bow = v_other.fit_transform(feats['other_talk'])
    #print(other_bow.shape)

    # Get exclusive editor non-unigram features
    ed_nonbow_d = {}
    for col in ed_nonbow_cols:
        ed_nonbow_d[col] = np.array([feats[col]]).T
    ed_nonbow = np.hstack(ed_nonbow_d.values())
    #print(ed_nonbow.shape)

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
    baseline()
    train_test(edbow, 'editor unigrams')
    train_test(ed_nonbow, 'editor CS')
    train_test(edfeats, 'editor unigrams+CS')
    train_test(bow_f, 'all unigrams')
    train_test(nonbow_f, 'all CS')
    train_test(feats_v, 'all unigrams+CS')


# # Logistic regression
# ## Leave-one-out CV

def train_test(train_data, desc):

    # Train and test logistic regression classifier
    clf = LogisticRegression(class_weight='balanced')
    #clf = LogisticRegression()
    loo = LeaveOneOut().split(train_data)
    scores = cross_validation.cross_val_score(clf, train_data, feats['editor_success'], cv=loo)
    print("%s Accuracy: %0.3f" % (desc, scores.mean()))

def baseline():
    print("Majority class guess {:0.3f} accuracy".format(
            max(1-sum(feats['editor_success'])/len(feats),
                sum(feats['editor_success']/len(feats)))))

def main():
    stats()
    eval()
    prediction()

if __name__ == '__main__':
    main()
