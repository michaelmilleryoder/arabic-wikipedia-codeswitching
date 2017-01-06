import csv
from string import ascii_letters
import re, hashlib, os
import sys

csv.field_size_limit(sys.maxsize)

# # See which revisions are reverts

dirpath = '/home/michael/school/research/wp/ar/ar_articles'
fnames = sorted(os.listdir(dirpath))
for i, fname in enumerate(fnames):
    if fname.endswith('.tsv'):
        revpath = os.path.join(dirpath, fname)
        reverts = []

        with open(revpath) as f:
            r = csv.reader(f, delimiter='\t') # be careful of delimiter
            revs = [row for row in r]

        hashes = set()
        for rev in revs:
            m = hashlib.md5()
            m.update(str(rev[2]).encode('utf8'))

            #restores previous version
            rv = (1 if m.digest() in hashes else 0)

            if not rv:
                hashes.add(m.digest())

            reverts.append(rv)    

        print("{:d} / {:d}".format(i, len(fnames)), end=": ")
        print(reverts.count(1))

        revs = [revs[i] + [reverts[i]] for i in range(len(revs))]

        with open('ar_articles/with_reverts/' + fname[:-4] + '.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(revs)
