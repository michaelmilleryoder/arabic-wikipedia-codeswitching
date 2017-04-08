import pandas as pd
import numpy as np
import sys

# # Filter full talk threads csv by scores

""" Functions """

def filter(cstalk_path, base_outpath, threshold=None):
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
            lose_threads.append((art,t))
            outpath = base_outpath
            
    mask = [tup in lose_threads for tup in zip(data['article'], data['thread_title'])]
    lose_rows = data[mask]

    #editor_len_mask = lose_rows['editor_talk'].map(lambda x: len(str(x).split()) > 0)
    #other_len_mask = lose_rows['other_talk'].map(lambda x: len(str(x).split()) > 0)

    editor_len_mask = lose_rows['editor_talk'].map(lambda x: isinstance(x, str))
    other_len_mask = lose_rows['other_talk'].map(lambda x: isinstance(x, str))

        # must have editor text in the contribution
    lose_rows = lose_rows[editor_len_mask]
    lose_rows = lose_rows[other_len_mask]
    print("Filtered #rows: {0}".format(len(lose_rows)))
    print("Filtered #threads: {0}".format(len(lose_threads)))

    lose_rows.to_csv(outpath, index=False)
    print("Wrote filtered threads to {0}".format(outpath))

    return outpath


def main():
    threshold = None
    cstalk_path = sys.argv[1]
    base_outpath = sys.argv[2]
    filter()


if __name__ == '__main__':
    main()
