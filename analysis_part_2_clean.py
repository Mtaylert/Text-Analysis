#Import all necessary packages
import analysis as at
import os
import codecs
import pandas as pd
import numpy as np
import spacy 
import pickle
import itertools as it
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
import os
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
import warnings
from gensim.models import Word2Vec
import datetime

#Create a dictionary of all the words the code has learned
trigram_dict_path = os.path.join(at.directory,'trigram_dct_all.dict')
#Define a function that only takes into account words that occur more than 5 
#times in the corpus but do not make up more than 30% of the words
def dic_tr(clean_revs_file):

    tri_rv = LineSentence(clean_revs_file)
    tri_dict = Dictionary(tri_rv)
    
    tri_dict.filter_extremes(no_below=5,no_above=0.3)
    tri_dict.compactify()
    tri_dict.save(trigram_dict_path)

#Load the dictionary back into the model    
tri_dictionary = Dictionary.load(trigram_dict_path)

#Define the path
tri_bag_path = os.path.join(at.directory,'tri_bag_all.mm')

#Define a function that creates a bag of words by taking each sentence in a review
#and stores it in a dictionary
def bow(filepath):
    for rev in LineSentence(filepath):
        yield tri_dictionary.doc2bow(rev)
        
#Make this if statement true (0 == 0) if you want to run code
#Assign a vector score to each word        
if 0 == 1:
    MmCorpus.serialize(tri_bag_path,
                       bow(at.trigram_reviews_path))

#Save the serialization done above to a pickle file and reading it back in    
trigram_bow_corpus_open = open(os.path.join(at.directory,"trigram_bow_corpus.pickle"),"rb")
trigram_bow_corpus = pickle.load(trigram_bow_corpus_open)
trigram_bow_corpus_open.close()

#Create a file to save the matrix created below
print('part 2 completed')
wvec_path = os.path.join(at.directory,'wrd_vec_model_all')

#Make a matrix out of the vectorized words
if 0 == 1:
#Make this if statement true (0 == 0) if you want to run code
    f_v = Word2Vec(at.trigram_sentences, size=100,window=5,
                        min_count=20,sg=1,workers=4)
    f_v.save(wvec_path)
    for i in range(1,12):
        f_v.train(at.trigram_sentences,total_examples=300,epochs=12)
        f_v.save(wvec_path)

#Loading the file back in
f_v = Word2Vec.load(wvec_path)

#Prints the number of words learned from the entire corpus
print(u'{:,} learned vocab words.'.format(len(f_v.wv.vocab)))

#Define function that returns top 'x' terms based on token search term
def rel_terms(token, topn=10):
    for w, s in f_v.wv.most_similar(positive=[token], topn=topn):
        print(u'{:20} {}'.format(w, round(s,3)))
        

#Define a function that adds and subtracts vector scores based on token search
#terms
def word_math(add=[], subtract=[],topn=1):
    answers = f_v.wv.most_similar(positive=add,negative=subtract,topn=topn)
    for w, s in answers:
        print(w)
        
#Examples 
word_list = [u'spicy',u'beer',u'pizza']
for i in word_list:
    print(rel_terms(i,topn=20))
    print(' ')



word_math(add=[u'lunch',u'breakfast'])
print(u' ')


word_math(add=[u'bun',u'mexican'],subtract=[u'american'])
print(u' ')

word_math(add=[u'coffee',u'snack'],subtract=[u'american',u'drink'])
print(u' ')

word_math(add=[u'night',u'drink'])
print(u' ')

print('done')
