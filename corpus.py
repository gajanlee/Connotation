import pandas as pd
import pickle
from tqdm import tqdm
import nltk
import numpy as np
import os


def datasets(selector):
    if selector.lower() == "marco":
        return _marco()

class Vocabulary:
    def __init__(self, vocab_list, vocab_dict, passage_count=None):
        self.vocab_list = vocab_list
        self.vocab_dict = vocab_dict
        self.UNKNOWN_ID, self.UNKNOWN_WORD = len(self.vocab_list), "<unknown>"
        self.passage_count = passage_count
        # self.vocab_list.append(self.UNKNOWN_WORD)
        # self.vocab_dict[self.UNKNOWN_WORD] = self.UNKNOWN_ID

    def get_vocab_id(self, word):
        return self.vocab_dict[word] if word in self.vocab_dict else self.UNKNOWN_ID
    
    def get_vocab_by_id(self, id):
        return self.vocab_list[id] if 0 <= id < len(self.vocab_list) else self.UNKNOW_WORD

# Convert words to count array
def convert_ids(vocab, words):
    counts = np.zeros((len(vocab.vocab_list)))
    for word in words:
        id = vocab.get_vocab_id(word)
        if id != vocab.UNKNOWN_ID: counts[id] += 1
    return counts

def _marco():
    marco_infile = "/media/home/lee/MARCO-dataset/train_v2.0_well_formed.json"
    marco_vocab_path = "marco_vocab.pkl"
    marco_data_path = "marco_data.npy"

    return marco_data(marco_infile, marco_vocab_path, marco_data_path)
     
"""
Vocabulary size is 1039964
less than 10 count is 926755
less than 20 count is 36934
less than 30 count is 16320
less than 40 count is 9412
less than 50 count is 6313
"""
def marco_data(marco_infile, marco_vocab_path, marco_data_path):
    df = pd.read_json(marco_infile)

    if os.path.exists(marco_vocab_path):
        with open(marco_vocab_path, "rb") as mv:
            vocab = pickle.load(mv)
    else:
        vocab_count, passage_count = {}, 0
        for _, row in zip(tqdm(range(len(df)), desc="generate vocabulary"), df.iterrows()):
            for passage in row[1]["passages"]:
                passage_count += 1
                for word in nltk.word_tokenize(passage["passage_text"]):
                    vocab_count[word] = vocab_count.get(word, 0) + 1
    
        vocab_list, vocab_dict = [], {}
        for w, c in vocab_count.items():
            if c <= 30: continue
            vocab_dict[w] = len(vocab_list)
            vocab_list.append(w)
        with open(marco_vocab_path, "wb") as mv:
            vocab = Vocabulary(vocab_list, vocab_dict, passage_count)
            pickle.dump(vocab, mv)
    
    # Read data and return numpy array
    if os.path.exists(marco_data_path):
        return (vocab.vocab_list, np.load(marco_data_path))
    
    passage_count = min(15000, vocab.passage_count)
    data = [] if vocab.passage_count is None else np.zeros((passage_count, len(vocab.vocab_list)), dtype="int8")        # Memory limit
    p_c = 0
    for _, row in zip(tqdm(range(passage_count), desc="generate count data"), df.iterrows()):
        if p_c == passage_count: break
        for _, passage in enumerate(row[1]["passages"]):
            id_counts = convert_ids(vocab, nltk.word_tokenize(passage["passage_text"]))
            if type(data) is list: data.append(id_counts)
            else: data[p_c] = id_counts
            p_c += 1; break
    data = np.array(data)
    np.save(marco_data_path, data)
    return (vocab.vocab_list, data)

datasets("MARCO")