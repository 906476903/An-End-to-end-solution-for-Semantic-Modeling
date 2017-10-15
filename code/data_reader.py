import numpy as np

import re
import codecs
import itertools
from collections import Counter
import json
article_len = 900

def load_test_data():
  test_examples = []
  input = open("./public/test.json","r")
  input = list(input)
  length = len(input)
  for i in range(length):
    dict = json.loads(input[i])
    test_examples += [dict["title"].strip() + dict["content"].strip()]

  print(len(test_examples))
  test_examples = [s.strip() for s in test_examples]
  
  test_examples_ = []
  for s in test_examples:
      p = list(s)
      test_examples_ += [p[:min(len(p),article_len)]]
  test_examples = test_examples_

  return test_examples
  
def load_data():
  pos = []
  neg = []
  input = list(open("./public/train.json","r"))
  length = len(input)
  for i in range(length):
    dict = json.loads(input[i])
    if(dict["label"] == '1'):
        pos += [dict["title"].strip() + dict["content"].strip()]
    else:
        neg += [dict["title"].strip() + dict["content"].strip()]
  print("pos: " + str(len(pos)))
  print("neg: " + str(len(neg)))
  articles = pos + neg
  articles_ = []
  for s in articles:
      p = list(s)
      articles_ += [p[:min(len(p),article_len)]]
  labels = np.concatenate([[[0, 1] for _ in pos], [[1, 0] for _ in neg]], 0)
  test_sentences = load_test_data()
  sentences_padded = []
  for i in range(len(test_sentences)):
    sentences_padded.append(test_sentences[i] + ["<PAD/>"] * (article_len - len(test_sentences[i])))
    
  test_sentences_padded = []
  for i in range(len(articles_)):
    test_sentences_padded.append(articles_[i] + ["<PAD/>"] * (article_len - len(articles_[i])))
    
  sentences = sentences_padded + test_sentences_padded
  counts = Counter(itertools.chain(*sentences))
  vocabulary_inv = [x[0] for x in counts.most_common()]
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  train_data = np.array([[vocabulary[word] for word in sentence] for sentence in sentences_padded])
  train_label = np.array(labels)
  test_data = np.array([[vocabulary[word] for word in sentence] for sentence in test_sentences_padded])
  return train_data, train_label, test_data, vocabulary


def batch_iter(data, batch_size, epochs):
  data = np.array(data)
  data_size = data.shape[0]
  num = int(data_size/batch_size) + 1
  for epoch in range(epochs):
    tot_batch = data[np.random.permutation(np.arange(data_size))]
    for batch_num in range(num):
      yield tot_batch[batch_num * batch_size : min((batch_num + 1) * batch_size, data_size)]
