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
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, mutual_info_regression, f_regression
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from scipy.sparse import hstack
from tqdm import tqdm
import numpy as np
import math
import sys
import pdb

"""
If run by itself, takes a command-line argument with the name of a feature file
(for example cs_talk_features.csv).

Prints experiment results.
"""


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

        # Save distributions
        np.save('{}_present.npy'.format(f), fpresent['editor_score'].values)
        np.save('{}_notpresent.npy'.format(f), nof['editor_score'].values)

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

    print("Vectorizing features ...")
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

    # Get editor Arabic unigram features
    #feats['editor_talk_arabic'].fillna('', inplace=True)
    #ed_arbow = v.fit_transform(feats['editor_talk_arabic'])

    ## Get editor Latin unigram features
    #feats['editor_talk_latin'].fillna('', inplace=True)
    #ed_latbow = v.fit_transform(feats['editor_talk_latin'])

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

    # Assemble editor features
    edfeats = hstack([edbow, ed_nonbow])

    # Assemble non-unigram features
    nonbow_f = np.hstack([ed_nonbow, nonbow])

    # Assemble unigram features
    bow_f = hstack([edbow, other_bow])

    # Assemble Latin unigram features and CS
    #cs_latbow = hstack([ed_latbow, ed_nonbow])

    # Assemble all features
    feats_v = hstack([edbow, other_bow, ed_nonbow, nonbow])

    # Train and test classifiers
    print("Training and testing classifiers ...")
    #u_scores = train_test(feats, edbow, 'editor unigrams')
    #print_top_features(v.get_feature_names(), LinearRegression().fit(edbow, feats['editor_score']), n=100)  

    scores = {}
    #print_top_features(ed_nonbow_cols, LinearRegression().fit(ed_nonbow, feats['editor_score']), n=len(ed_nonbow_cols))  

    #train_test(feats, edfeats, 'editor unigrams+CS')
    feat_names_ed = v.get_feature_names() + ed_nonbow_cols
    #print_top_features(feat_names_ed, LinearRegression().fit(edfeats, feats['editor_score']), n=100)  

    # Train and test with different feature sets
    clf_type = LinearRegression()
    #clf_type = svm.SVR()
    feat_nums = [10]
    featset_list = {}

    # Feature sets that require feature selection
    featset_list['featsel'] = [
        #(edbow, 'editor unigrams'),
        #(edfeats, 'editor unigrams+CS'),
        #(bow_f, 'editor+other unigrams'),
        #(feats_v, 'editor+other unigrams+CS')
        ] 

    for p in feat_nums:
        tqdm.write("\n******{} FEATURES******".format(p))
        for featset, desc in tqdm(featset_list['featsel']):
            scores[desc] = train_test(feats, featset, desc, initial_clf=clf_type, feat_selection='kbest', num_feats=p, save=True)

            # t-test for significance between unigrams and cs MSE
            if desc != 'editor unigrams':
                tqdm.write('T-test between unigrams and {}: {}'.format(desc, ttest_ind(scores['editor unigrams'], scores[desc])))

    # Feature sets that don't require feature selection
    featset_list['no_featsel'] = [
        #(cs_latbow, 'editor Latin unigrams+CS')
        #(ed_nonbow, 'editor CS'),
        #(nonbow_f, 'editor+other CS')        
        ]

    for featset, desc in featset_list['no_featsel']:
        scores[desc] = train_test(feats, featset, desc, save=True)
        tqdm.write('T-test between unigrams and {}: {}\n'.format(desc, ttest_ind(scores['editor unigrams'], scores[desc])))

    # Print most informative features for unigrams
    num_feats = feat_nums[0]
    print("\nMost informative {} unigrams features...".format(num_feats))
    print_top_features_selected(edbow, v.get_feature_names(), num_feats, feats['editor_score'], clf_type, n=min(num_feats,20))

    # Print most informative features for unigrams+CS
    #print("\nMost informative {} unigrams+CS features...".format(num_feats))
    #print_top_features_selected(edfeats, feat_names_ed, num_feats, feats['editor_score'], clf_type, n=20)
    

def train_test(feats, train_data, desc, initial_clf=LinearRegression(), feat_selection=None, num_feats=1, print_top_features=None, featnames=None, save=False):

    # Baseline for classification
    #baseline(feats)

    # Train and test logistic regression classifier
    #clf = LogisticRegression(class_weight='balanced')
    #clf = LogisticRegression()
    ##loo = LeaveOneOut().split(train_data)
    #scores = cross_val_score(clf, train_data, feats['editor_success'], cv=10)
    #print("%s Accuracy: %0.3f" % (desc, scores.mean()))

    # Train and test linear regression classifier
    clf = initial_clf # overwritten if feature selection

    if feat_selection:

        if feat_selection == 'percentile':
            selector = SelectPercentile(mutual_info_regression, percentile=num_feats)
        elif feat_selection == 'kbest':
            selector = SelectKBest(mutual_info_regression, k=num_feats)
        else: 
            print("Only percentile or kbest feature selection allowed")
            return

        clf = make_pipeline(selector, initial_clf)

    pred = cross_val_predict(clf, train_data, feats['editor_score'], cv=10, n_jobs=-1)
    #pred = cross_val_predict(clf, train_data, feats['editor_score'], cv=10)

    pred = pred.clip(0,1) # clip between 0 and 1
    rmse = math.sqrt(mean_squared_error(feats['editor_score'], pred))
    print("%s RMSE: %0.3f" % (desc, rmse))

    if save:
        with open("{}.csv".format(desc.replace(' ', '_').lower()), 'w') as f:
            for p in pred:
                f.write("{:f}\n".format(p))

    errors = np.sqrt((feats['editor_score'] - pred) ** 2)
    return errors # for significance testing

    # SVM classifier
    #clf = svm.SVC()
    #scores = cross_val_score(clf, train_data, feats['editor_success'], cv=10)
    #print("%s Accuracy: %0.3f" % (desc, scores.mean()))


def baseline(feats):
    # right now just for classification
    print("Majority class guess {:0.3f} accuracy".format(
            max(1-sum(feats['editor_success'])/len(feats),
                sum(feats['editor_success']/len(feats)))))

def print_top_features_selected(featset, total_featnames, num_feats_selected, y, clf, n=20):
    """ Prints features with the highest coefficient values,
        with feature selection """
    selector = SelectKBest(mutual_info_regression, k=num_feats_selected).fit(featset, y)
    #feat_scores = selector.scores_
    X_new = selector.transform(featset)
    featnames = [tup[0] for tup in zip(total_featnames, selector.get_support()) if all(tup)]
    print_top_features(featnames, clf.fit(X_new, y), n=n)

def print_top_features(feat_names, clf, n=20):
    """ Prints features with the highest coefficient values """
    top = np.argsort(clf.coef_)[-1*n:]
    print(" ".join(reversed([feat_names[j] for j in top])))
    print()

def main():
    featfile = sys.argv[1] # input file, such as cs_talk_features.csv
    
    # Load features
    feats = pd.read_csv(featfile)

    stats(feats)
    evaluate(feats)
    prediction(feats)

if __name__ == '__main__':
    main()
