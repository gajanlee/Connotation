import pandas as pd
from tqdm import tqdm
import nltk

class Vocabulary:
    def __init__(self, vocab_list, vocab_dict):
        self.vocab_list = vocab_list
        self.vocab_dict = vocab_dict
        self.UNKNOW_ID, self.UNKNOW_WORD = len(self.vocab_list), "<unknown>"
        # self.vocab_list.append(self.UNKNOW_WORD)
        # self.vocab_dict[self.UNKNOW_WORD] = self.UNKNOW_ID

    def get_vocab_id(self, word):
        return self.vocab_dict[word] if word in self.vocab_dict else self.UNKNOW_ID
    
    def get_vocab_by_id(self, id):
        return self.vocab_list[id] if 0 <= id < len(self.vocab_list) else self.UNKNOW_WORD

"""
Vocabulary size is 1039964
less than 10 count is 926755
less than 20 count is 36934
less than 30 count is 16320
less than 40 count is 9412
less than 50 count is 6313
"""
def marco_reader():
    infile = "/media/home/lee/MARCO-dataset/train_v2.0_well_formed.json"
    df = pd.read_json(infile)
    vocab_count = {}
    with tqdm(total=len(df)) as pbar:
        for row in df.iterrows():
            pbar.update(1)
            for passage in row[1]["passages"]:
                for word in nltk.word_tokenize(passage["passage_text"]):
                    vocab_count[word] = vocab_count.get(word, 0) + 1
    print("vocab size is {}".format(len(vocab_count)))
    vocab_list, vocab_dict = [], {}
    for w, c in vocab_count.items():
        if c <= 30: continue
        vocab_dict[w] = len(vocab_list)
        vocab_list.append(w)
    with open("vocab.pkl", "wb") as vp:
        pickle.dump(Vocabulary(vocab_list, vocab_dict), vp)
marco_reader()