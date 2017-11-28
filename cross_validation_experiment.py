from sklearn.cross_validation import StratifiedKFold
from fasta_reader import get_dataset
from conv_model import build_conv_lstm
import numpy as np
from keras.preprocessing.text import one_hot

data_file = 'data/16S.fas'
taxonomy_file = 'data/taxonomy.csv'
number_of_columns = 5

for run in range(number_of_columns):
    ids, seqs, labels = get_dataset(data_file, taxonomy_file, run+1)
    unique_labels = list(set(labels))
    print(len(unique_labels))
    #seqs = [" ".join(list(seq)) for seq in seqs]
    #y = np.array([unique_labels.index(x) for x in labels])
    #print("y: " + str(y))
   # print(one_hot(str(seqs[0]), 4))
  #  print(seqs[0])
 #   x = [np.array(one_hot(seq, 4, filters="")) for seq in str(seqs)]
    #print(len(x))
    i = 0
    """for train_idx, test_idx in StratifiedKFold(y, 10, shuffle=True):
        model = build_conv_lstm(3)
        model.fit(seqs[train_idx], y[train_idx], batch_size=100, verbose=1)
        out = model.predict_classes(seqs[test_idx])
        np.save(str(run) + "_cv" + str(i)+".txt", 'w')
        i = i+1
        print(model.evaluate(seqs[test_idx], y[test_idx]))"""


