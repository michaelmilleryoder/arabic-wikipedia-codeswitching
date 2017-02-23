from filter_threads import filter
from extract_cs_features import extract_features
from experiments import stats, evaluate
import pandas as pd
import numpy as np

#threshold = None

def main():

    for threshold in np.linspace(0.5, 0.9, 5):

        print("************* THRESHOLD {0} *****************".format(threshold))

        # filter threads
        print("Filtering threads with threshold {0} ...".format(threshold))
        filtered_fpath = filter(cstalk_path='../cs_talk_scores.csv',
                            base_outpath='../cs_talk_filtered.csv',
                            threshold=threshold)
        print()

        # extract cs features
        print("Extracting code-switching features ...")
        features_fpath = filtered_fpath.replace('filtered', 'features')
        extract_features(filtered_fpath, features_fpath)
        print()

        # run experiments
        print("Running experiments ...")
        feats = pd.read_csv(features_fpath)
        stats(feats)
        evaluate(feats)

if __name__ == '__main__':
    main()
