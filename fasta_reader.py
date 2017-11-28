from Bio import SeqIO
import numpy as np


def get_nucleosome_dataset(species):
    assert species in ["elegans", "melanogaster", "sapiens"]
    prefix = "res/fasta_files/nucleosomes_vs_linkers_"
    handle = open(prefix + species + ".fas", "rU")
    seqs = []
    labels = []
    for record in SeqIO.parse(handle, "fasta"):
        seqs.append(record.seq)
        labels.append(record.id)
    handle.close()
    return np.array(seqs), np.array(labels)

def get_dataset(sequence_file, label_file, column):
    handle = open(sequence_file, "rU")
    seqs = []
    ids = []
    labels = []
    taxonomy = get_taxonomy(label_file, column)
    for record in SeqIO.parse(handle, "fasta"):
        seq_id = record.id
        seqs.append(record.seq)
        ids.append(seq_id)
        labels.append(taxonomy[seq_id])
    handle.close()
    return np.array(ids), np.array(seqs), np.array(labels)

def get_taxonomy(file, column):
    """Column start from 1"""
    tax = dict()
    with open(file, 'r') as fin:
        for line in fin:
            tokens = line.strip().split(',')
            tax[tokens[0]] = tokens[column]
    return tax



if __name__ == '__main__':
    print(get_dataset('data/16S.fas', 'data/taxonomy.csv', 1))
    #print(get_taxonomy('data/taxonomy.csv', 1))
