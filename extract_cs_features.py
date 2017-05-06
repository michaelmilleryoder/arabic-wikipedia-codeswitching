import pandas as pd
import csv, pdb
from string import ascii_letters, punctuation
letters = ascii_letters + punctuation
from collections import Counter, defaultdict
import re, hashlib, os, sys
import numpy as np
import math


""" utility fns """
def latin_named_entities(text):
    """ Returns the proportion of entities in Latin characters in a text"""
    lat_wds = get_latin(text).split()
    if len(lat_wds) == 0:
        return 0.0
    caps = sum(1 for w in lat_wds if w[0].isupper())
    return caps/len(lat_wds)

# ## quoting
# Must also have code-switching
def two_quotes(text):
    return str(text).count('"') >= 2

# ## proportion of switches between Latin and non-Latin words
# Doesn't have to have code-switching
def prop_switches(text):
    wds = [all_latin(w) for w in str(text).split()]
    n_switches = sum(wds[i] != wds[i+1] for i in range(len(wds)-1))
    if len(wds) > 1:
        return n_switches/(len(wds)-1)
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

# See if a word is all Latin characters
def all_latin(word):
    if len(word) > 1 and all(c in letters for c in word) \
            and any(c in ascii_letters for c in word) \
            and not word.startswith('http') \
            and word != '(UTC)' \
            and word != '\\n' \
            and not 'TEMPLATE' in word \
            and not 'IPADDRESS' in word \
            and not 'TIME' in word:
        return True
    else:
        return False

def all_arabic(wd):
    """ Returns true if all characters are Arabic in a word, else false """
    if len(wd) <= 1:
        return False

    all_ar = False
    for c in wd:
        val = ord(c)
        if not (val > int('0x600', 16) and val < int('0x6ff', 16)):
            break
        else:
            all_ar = True
    
    return all_ar


# See if a text has at least three words with Latin letters longer than 1 letter
def has_latin(text):
    if sum(all_latin(w) for w in str(text).split()) > 2:
        return True
    else:
        return False

# Proportion of words with all Latin characters
def prop_latin(text):
    if len(str(text).split()) == 0:
        return 0.0

    else:
        n_wds = sum(all_latin(w) for w in str(text).split())
        return n_wds/len(str(text).split())

# Return words with Latin letters longer than 1 letter
def get_latin(text):
    latin_wds = [w for w in str(text).split() if all_latin(w)]
    if len(latin_wds) > 2:
        return ' '.join(latin_wds)
    else:
        return None

def get_arabic(text):
    """ Return words with Arabic letters longer than 1 letter """

    arabic_wds = [w for w in str(text).split() if all_arabic(w)]
    if len(arabic_wds) > 0:
        return ' '.join(arabic_wds)
    else:
        return None

""" extraction functions """
def extract_named_entities(feats):
    ed_crit = [0.0 if not a and b else a for a,b in zip(feats['editor_talk'].map(latin_named_entities), feats['en_cs'].tolist())]
    other_crit = [0.0 if not a and b else a for a,b in zip(feats['other_talk'].map(latin_named_entities), feats['other_en_cs'].tolist())]

    feats['editor_latin_named_entities'] = ed_crit
    feats['other_latin_named_entities'] = other_crit

def extract_quotes(feats):
    """ Two quote marks and presence of Latin words """
    ed_crit = [a and b for a,b in zip(feats['editor_talk'].map(two_quotes), feats['latin_cs'].tolist())]
    other_crit = [a and b for a,b in zip(feats['other_talk'].map(two_quotes), feats['other_latin_cs'].tolist())]

    feats['editor_two_quotes'] = ed_crit
    feats['other_two_quotes'] = other_crit

def extract_prop_latin_switches(feats):
    """ Proportion of switches between Latin and non-Latin words """
    ed_crit = feats['editor_talk'].map(prop_switches)
    other_crit = feats['other_talk'].map(prop_switches)
    feats['editor_prop_switches'] = ed_crit
    feats['other_prop_switches'] = other_crit

def extract_prop_english(feats):
    crit = feats['editor_talk'].map(prop_english)
    feats['editor_prop_en'] = crit
    crit = feats['other_talk'].map(prop_english)
    feats['other_prop_en'] = crit

# TODO: Put in lang identification
def extract_has_english(feats):
    # Posts with code-switching
    crit = feats['editor_talk'].map(has_english)
    feats['en_cs'] = crit
    crit = feats['other_talk'].map(has_english)
    feats['other_en_cs'] = crit
    print("Number of posts with English: {0} / {1}".format(sum(crit), len(feats)))
    print("Number of other posts with English: {0} / {1}".format(sum(crit), len(feats)))

def extract_has_latin(feats):
    # Posts with code-switching
    feats['latin_cs'] = feats['editor_talk'].map(has_latin)
    feats['other_latin_cs'] = feats['other_talk'].map(has_latin)
    print("Number of posts with Latin: {0} / {1}".format(
            sum(feats['latin_cs']), len(feats)))
    print("Number of other posts with Latin: {0} / {1}".format(
            sum(feats['other_latin_cs']), len(feats)))

def extract_prop_latin(feats):
    crit = feats['editor_talk'].map(prop_latin)
    feats['editor_prop_latin'] = crit
    crit = feats['other_talk'].map(prop_latin)
    feats['other_prop_latin'] = crit

def extract_arabic_words(feats):
    feats['editor_talk_arabic'] = feats['editor_talk'].map(get_arabic)

def extract_latin_words(feats):
    feats['editor_talk_latin'] = feats['editor_talk'].map(get_latin)

def get_editor_success(feats):
    feats['editor_success'] = feats['editor_score'].map(lambda x: x > 0.5)


def extract_features(featfile, outfile):
    """ Main extraction function """

    feats = pd.read_csv(featfile)
    print("Extracting Latin features...")
    extract_has_latin(feats)
    extract_prop_latin(feats)
    extract_prop_latin_switches(feats)
    extract_latin_words(feats)

    print("Extracting Arabic features ...")
    extract_arabic_words(feats)

    print("Extracting quote features...")
    extract_quotes(feats)
    extract_named_entities()
    get_editor_success(feats)
    feats.to_csv(outfile, index=False)
    print("Wrote features to {0}".format(outfile))


def main():
    """ For use with command-line """

    """ I/O definitions """
    featfile = sys.argv[1] # input file, ar/cs_talk_filtered.csv
    outfile = sys.argv[2] # cs_talk_features.csv

    #extract_has_english()
    #extract_prop_english()
    extract_features(featfile, outfile)

if __name__ == '__main__':
    main()
