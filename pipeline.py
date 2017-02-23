from filter_threads import filter
import extract_cs_features
import experiments

threshold = None

def main():
    # filter threads
    print("Filtering threads with threshold {0}".format(threshold))
    filter(cstalk_path='../cs_talk_scores.csv',
                        base_outpath='../cs_talk_filtered.csv')

if __name__ == '__main__':
    main()
