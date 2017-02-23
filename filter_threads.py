import pandas as pd
import numpy as np
import sys

# # Filter full talk threads csv with scores

#""" I/O and settings """

#cstalk_path = sys.argv[1] # ar/cs_talk_scores.csv
#base_outpath = sys.argv[2] # ar/cs_talk_filtered.csv
#threshold = 0.5 # threads can't have all editors scoring above this value


""" Functions """

def filter(threshold, cstalk_path, base_outpath):
    # ## Filter out conversations where everyone wins or only 1 editor present

    data = pd.read_csv(cstalk_path)
    print("Original #rows: {0}".format(len(data)))

    threads = sorted(set(zip(data['article'], data['thread_title'])))
    print("Original #threads: {0}".format(len(threads)))

    lose_threads = []

    for art, t in threads:
        t_rows = data[(data['article']==art) & (data['thread_title']==t)]
        if len(t_rows) <= 1:
            continue
        scores = t_rows['editor_score'].tolist()
        if threshold:
            if not all([s > threshold for s in scores]):
                lose_threads.append((art,t))
            outpath = "{0}_{1}.csv".format(base_outpath[:-4], threshold)
        else: 
            outpath = base_outpath
            
    print("Filtered #threads: {0}".format(len(lose_threads)))

    mask = [tup in lose_threads for tup in zip(data['article'], data['thread_title'])]
    lose_rows = data[mask]
    print("Filtered #rows: {0}".format(len(lose_rows)))

    lose_rows.to_csv(outpath, index=False)


def main():
    threshold = None
    cstalk_path = sys.argv[1]
    base_outpath = sys.argv[2]
    filter()


if __name__ == '__main__':
    main()
