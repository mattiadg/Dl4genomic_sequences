from fasta_reader import get_dataset
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import itertools

char2arr = {'a' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'c' : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'g' : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            't' : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'n' : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            's' : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'r' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'y' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'k' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'm' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'w' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'd' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'b' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'h' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'v' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'z' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

charmap = {'a' : ['a'],
           'c' : ['c'],
           'g' : ['g'],
           't' : ['t'],
           'r' : ['a', 'g'],
           'y' : ['c', 't'],
           's' : ['g', 'c'],
           'w' : ['a', 't'],
           'k' : ['g', 't'],
           'm' : ['a', 'c'],
           'b' : ['c', 'g', 't'],
           'd' : ['a', 'g', 't'],
           'h' : ['a', 'c', 't'],
           'v' : ['a', 'g', 'c'],
           'n' : ['a', 'c', 'g', 't']
}

data_file = 'data/16S.fas'
taxonomy_file = 'data/taxonomy.csv'
number_of_columns = 5

def vcount(vec):
    unique, counts = np.unique(vec, return_counts=True)
    return dict(zip(unique, counts))


if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)
    k = 3

    for run in range(number_of_columns):
        print("run " + str(run))
        ids, seqs, labels = get_dataset(data_file, taxonomy_file, run+1)
        unique_labels = list(set(labels))
        y = np.array([unique_labels.index(x) for x in labels])
        #Convert sequences of characters to sequences of numbers
        #seqs = np.array(list(map(lambda s: np.array(list(map(lambda c: np.array(char2arr[c]), str(s)))), new_seqs)))

        #Remove non-nucleotides characters
        #nseqs = np.array(list(map(lambda seq: "".join(map(lambda ch: charmap[ch][np.random.randint(0, len(charmap[ch]))], seq)), seqs)))

        #Convert to kmers
        kmers = list(itertools.product(['a', 'c', 'g', 't'], repeat=k))
        kmers_d = np.zeros((3000, 4**k), dtype=float)
        count = 0
        """for seq in nseqs:
            seq = str(seq)
            for i in range(len(seq)-k+1):
                idx = kmers.index(tuple(seq[i:i+k]))
                kmers_d[count, idx] += 1
            kmers_d[count, :] = np.divide(kmers_d[count, :], kmers_d[count, :].sum())
            count += 1
        """
        print(y)
        print(vcount(y))
        #y = np_utils.to_categorical(y, max(y)+1)
        test_predicted1 = np.zeros((labels.shape[0],))
        test_predicted2 = np.zeros((labels.shape[0],))
        i=0
        #For each one of the 10 fold, fit the classifier and test
        for train_idx, test_idx in StratifiedKFold(y, 10, True):
            i += 1
            print("fold " + str(i))
            #train_idx = get_balanced_classes(train_idx, np.argmax(y, axis=1))
            X_train, X_test = kmers_d[train_idx, :], kmers_d[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]
            print(set(train_idx).intersection(set(test_idx)))
            print(vcount(y_test))
