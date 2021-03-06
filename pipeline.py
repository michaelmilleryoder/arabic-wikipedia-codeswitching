from filter_threads import filter
from extract_cs_features import extract_features
from experiments import stats, evaluate
import pandas as pd
import numpy as np

""" Pipeline for filtering threads, extracting features and running experiments.
Requires no command-line arguments. """

#threshold = None

def main():

    #for threshold in np.linspace(0.5, 0.9, 5):
    threshold = None

    print("************* THRESHOLD {0} *****************".format(threshold))

    # filter threads
    #print("Filtering threads with threshold {0} ...".format(threshold))
    #filtered_fpath = filter(cstalk_path='../cs_talk_scores.csv',
    #                    base_outpath='../cs_talk_filtered.csv',
    #                    threshold=threshold)
    #print()

    # extract cs features
    filtered_fpath = '../cs_talk_filtered.csv' # if already filtered

    print("Extracting code-switching features ...")
    features_fpath = filtered_fpath.replace('filtered', 'features')
    extract_features(filtered_fpath, features_fpath)
    print()

    # run experiments
    print("Running experiments ...")
    feats = pd.read_csv(features_fpath)
    stats(feats)
    evaluate(feats) # just statistical analysis, no classification

if __name__ == '__main__':
    main()
