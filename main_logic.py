from conv_model import build_lstm
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from fasta_reader import get_dataset

charmap = {'a' : 1,
           'c' : 2,
           'g' : 3,
           't' : 4,
           'r' : 5,
           'y' : 6,
           's' : 7,
           'w' : 8,
           'k' : 9,
           'm' : 10,
           'b' : 11,
           'd' : 12,
           'h' : 13,
           'v' : 14,
           'n' : 15
}

number_of_columns = 5

if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)

    #kmers_d = np.load('models/5-mers.npy', allow_pickle=True)
    labels = np.load('models/labels.npy', allow_pickle=True)
    _, seqs, _ = get_dataset('data/16S.fas', 'data/taxonomy.csv', 1)
    new_seqs = np.array([" ".join(list(seq)) for seq in seqs])
    new_seqs = np.zeros((3000, 1573))
    for i in range(seqs.shape[0]):
        for j in range(len(seqs[i])):
            new_seqs[i, j] = charmap[seqs[i][j]]
    for run in range(number_of_columns):
        k = 5

        lb = labels[run, :]
        y = np_utils.to_categorical(lb, max(lb)+1)
        test_predicted = np.zeros((y.shape[0],))
        i=0
        #For each one of the 10 fold, fit the classifier and test
        for train_idx, test_idx in StratifiedKFold(lb, 10, True):
            i += 1
            print("FOLD " + str(i))
            #train_idx = get_balanced_classes(train_idx, np.argmax(y, axis=1))
            #X_train, X_test = seqs[train_idx, :, :], seqs[test_idx, :, :]
            X_train, X_test = new_seqs[train_idx, :], new_seqs[test_idx, :]
            y_train, y_test = y[train_idx, :], y[test_idx, :]
            #Build new network
            model = build_lstm(20, y[0].shape[0])
            #model = build_conv_lstm(3, max_len, 16, y[0].shape[0])
            #model = build_cnn(y[0].shape[0])
            early = EarlyStopping(patience=3)
            mcp = ModelCheckpoint('iter' + str(i) +".model", save_best_only=True)
            model.fit(X_train, y_train, nb_epoch=15, batch_size=64, verbose=1)
            test_predicted[test_idx] = np.argmax(model.predict(X_test), axis=1)
            print(accuracy_score(np.argmax(y_test, axis=1), test_predicted[test_idx]))

        np.save('predictions_BIGMODEL_' + str(run), test_predicted)
        print("Column " + str(run+1) + " " + str(accuracy_score(np.argmax(y, axis=1), test_predicted)))
