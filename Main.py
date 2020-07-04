import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm
import logging
from Model import Model

def nCr(n,r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))

def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc

def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns); cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1,w2] > 0):
            word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]}))
    return word_word

def Generate_data(window=10):
  Train = pd.read_csv('D:/Kenya/Train.csv')
  Test = pd.read_csv('D:/Kenya/Test.csv')
  Submission = pd.read_csv('D:/Kenya/SampleSubmission.csv')
  


  A = list(Train.label.unique())
  for i in range(len(Train)):
    Train.label[i] = A.index(Train.label[i])
  Train.sort_values(by='label',inplace = True)

  Train = Train.reset_index(drop=True)


  Test['label'] = 4
  ind = []
  for i in range(len(Train),len(Train)+len(Test)):
    ind.append(i)
  Test['index'] = 0
  for i in range(len(Test)):
    Test['index'][i] = ind[i]
  Test.set_index('index',inplace=True)

  con = [Train,Test]
  Data = pd.concat(con)
  

  stopwords = list(set(nltk.corpus.stopwords.words("english")))
  
  Data['text'] = Data['text'].apply(lambda x:nltk.word_tokenize(x)).apply(lambda x:filter_tokens(x,stopwords))    
  vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
  vectorizer.fit(Data["text"])
  df_tfidf = vectorizer.transform(Data['text'])
  df_tfidf = df_tfidf.toarray()
  vocab =  vectorizer.get_feature_names()
  vocab = np.array(vocab)
  df_tfidf = pd.DataFrame(df_tfidf,columns=vocab)
  names = vocab
  n_i = OrderedDict((name,0) for name in names)
  word2index = OrderedDict((name,index) for index,name in enumerate(names))
  occurrences = np.zeros( (len(names),len(names)) ,dtype=np.int32)
  no_windows = 0
  for l in tqdm(Data["text"], total=len(Data["text"])):
        for i in range(len(l)-window):
            no_windows += 1
            d = set(l[i:(i+window)])
            for w in d:
                n_i[w] += 1
            for w1,w2 in combinations(d,2):
                i1 = word2index[w1]
                i2 = word2index[w2]
                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1
  p_ij = pd.DataFrame(occurrences, index = names,columns=names)/no_windows
  p_i = pd.Series(n_i, index=n_i.keys())/no_windows
  for col in p_ij.columns:
        p_ij[col] = p_ij[col]/p_i[col]
  for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:]/p_i[row]
  p_ij = p_ij + 1E-9
  for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
  

  G = nx.Graph()
  G.add_nodes_from(df_tfidf.index)
  G.add_nodes_from(vocab)
  document_word = [(doc,w,{"weight":df_tfidf.loc[doc,w]}) for doc in tqdm(df_tfidf.index, total=len(df_tfidf.index))\
                     for w in df_tfidf.columns]
  word_word = word_word_edges(p_ij)
  G.add_edges_from(document_word)
  G.add_edges_from(word_word)
  Mod = Model(G,Data,ind,Submission,A)


if __name__=="__main__":
  Generate_data()