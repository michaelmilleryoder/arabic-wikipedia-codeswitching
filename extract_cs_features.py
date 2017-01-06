import pandas as pd
import csv, pdb
from string import ascii_letters
from nltk.corpus import words, stopwords
import nltk
from collections import Counter, defaultdict
import re, hashlib, os, sys
endict = set(words.words())
from pandas.tseries.offsets import *
#from sklearn.metrics import cohen_kappa_score, make_scorer
#from sklearn import cross_validation, svm
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import train_test_split, LeaveOneOut
#from scipy.stats import ttest_rel, ttest_ind, pearsonr
#from scipy.sparse import hstack
import numpy as np
import math


""" I/O definitions """
#featfile = '/home/michael/school/research/wp/ar/discussion_features.csv'
featfile = sys.argv[1]
feats = pd.read_csv(featfile)




# ## technical terms in En
# Doesn't have to have code-switching

""" utility fns """
def en_technical(text):
#     if not has_english(text): return 0.0
    en_wds = [w for w in str(text).split() if len(w) > 1 and w.lower() in endict and w != 'TEMPLATE']
    if len(en_wds) == 0:
        return 0.0
    caps = sum(1 for w in en_wds if w[0].isupper())
    return caps/len(en_wds)

# ## quoting
# Must also have code-switching
def two_quotes(text):
    return str(text).count('"') >= 2

# ## proportion of switches
# Doesn't have to have code-switching
def prop_switches(text):
    en = [w in endict for w in str(text).split() if len(w) > 1]
    n_switches = sum(en[i] != en[i+1] for i in range(len(en)-1))
    if len(en) > 1:
        return n_switches/(len(en)-1)
    else:
        return 0.0

# Proportion of English words
def prop_english(text):
    n_en = sum(w in endict for w in str(text).split() if len(w) > 1)
    return n_en/len(str(text).split())

# See if a text has at least three English words (don't have to be unique)
def has_english(text):
    if sum(w in endict for w in str(text).split() if len(w) > 1) > 2:
        return True
    else:
        return False




""" fns to run """
def extract_technical():
    ed_crit = [0.0 if not a and b else a for a,b in zip(feats['editor_talk'].map(en_technical), feats['en_cs'].tolist())]
    other_crit = [0.0 if not a and b else a for a,b in zip(feats['other_talk'].map(en_technical), feats['other_en_cs'].tolist())]

    feats['editor_en_technical'] = ed_crit
    feats['other_en_technical'] = other_crit

def extract_quotes():
    ed_crit = [a and b for a,b in zip(feats['editor_talk'].map(two_quotes), feats['en_cs'].tolist())]
    other_crit = [a and b for a,b in zip(feats['other_talk'].map(two_quotes), feats['other_en_cs'].tolist())]

    feats['editor_two_quotes'] = ed_crit
    feats['other_two_quotes'] = other_crit

def extract_prop_switches():
    ed_crit = feats['editor_talk'].map(prop_switches)
    other_crit = feats['other_talk'].map(prop_switches)
    feats['editor_prop_switches'] = ed_crit
    feats['other_prop_switches'] = other_crit

def extract_prop_english():
    crit = feats['editor_talk'].map(prop_english)
    feats['editor_prop_en'] = crit
    crit = feats['other_talk'].map(prop_english)
    feats['other_prop_en'] = crit

def extract_has_english():
    # Posts with code-switching
    crit = feats['editor_talk'].map(has_english)
    feats['en_cs'] = crit
    crit = feats['other_talk'].map(has_english)
    feats['other_en_cs'] = crit
    print("Number of posts with English: {0} / {1}".format(sum(crit), len(feats)))
    print("Number of other posts with English: {0} / {1}".format(sum(crit), len(feats)))

def get_editor_success():
    feats['editor_success'] = feats['editor_score'].map(lambda x: x > 0.5)


def main():

    extract_has_english()
    extract_prop_english()
    extract_prop_switches()
    extract_quotes()
    extract_technical()
    get_editor_success()
    feats.to_csv(featfile, index=False)
    print("Wrote features")

if __name__ == '__main__':
    main()
