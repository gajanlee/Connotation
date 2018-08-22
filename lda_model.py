import lda
import numpy as np

from corpus import datasets

vocab_list, data = datasets("MARCO")
print("load data done")
model = lda.LDA(n_topics=300, n_iter=1500, random_state=1)
print("model")
model.fit(data)
print("training")
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab_list)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
