#Imported necessary packages.
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
import pyLDAvis
import pyLDAvis.gensim
import warnings
from gensim.models import Word2Vec
import datetime

#Read in Yelp reviews
yelp = pd.read_csv('yelp.csv')
yelp['business_categories'] = yelp['business_categories'].fillna('remove')
df = yelp[yelp['business_categories'].str.contains('Restaurants')]

#Set working directory to where the data and files are stored.
directory  = 'C:\\Users\\e2slp2f\\.spyder-py3\\new_text_project\\'

#Load 'spacy' in to read words in English.
nlp = spacy.load('en')

#Define function to split data into training and test data.
def test_vs_train(df,col):
    split = int(len(df[col])*.70)
    train = df[:split]
    test = df[split:]
    split_data = {'train':train,'test':test}
    return split_data

#Split the data using previously defined function.
split_data = test_vs_train(df,'text')
training = split_data['train']
test = split_data['test']


#Define function in order to identify punctuation and spaces.
def punct_space_removal(token):
    return token.is_punct or token.is_space

#Define a function that normalizes basic text, removes punctuation, spaces and 
#stem of words. Function also tuples the normalized data with the original 
#data as to continue carrying forward the stars, original text, and business 
#name.Return normalized data frame.
def normalize(df,col,col2,col3):
    norms = []
    for a,b,c in zip(df[col],df[col2],df[col3]):
        try:
            parsed = nlp(str(a))
        except:
            pass
        for num,sentence in enumerate(parsed.sents):
            norms.append(tuple([u' '.join([token.lemma_ for token in sentence
                                    if not punct_space_removal(token)]),b,c]))
    norms_df = pd.DataFrame(norms)
    norms_df.columns = ['Normalized_Text','Stars','Business_Name']
    return norms_df

#Read in a 'pickled' file.          
train_open = open(os.path.join(directory,"normalize.pickle"),"rb")
train_new = pickle.load(train_open)
train_open.close()


text = train_new['Normalized_Text']

#Creates a file containing first part of phrasing.
unigram_sent_path = os.path.join(directory,'unigram.txt')

if 0 ==1:
#Make this if statement true (0 == 0) if you want to run code.
    with codecs.open(unigram_sent_path,'w',encoding='utf-8') as f:
        for sentence in text:
            f.write(sentence + '\n')
            
#Reading in more 'pickle' files.
uni_open = open(os.path.join(directory,"unigram.pickle"), "rb")
uni_sentence = pickle.load(uni_open)
uni_open.close()

bigrm_mdl_path = os.path.join(directory,'bigram.txt')

if 0 == 1:
#Make this if statement true (0 == 0) if you want to run code.
    bigram_model = Phrases(uni_sentence)
    bigram_model.save(bigrm_mdl_path)
    
#More pickles!    
bigram_open = open(os.path.join(directory,"bigram.pickle"),"rb")
bigram_model = pickle.load(bigram_open)
bigram_open.close()

bigrm_sentences_fp = os.path.join(directory,'bigrm_sentences_all.txt')

if 0 == 1:
#Make this if statement true (0 == 0) if you want to run code.
    with codecs.open(bigrm_sentences_fp,'w',encoding='utf_8')as f:
        for uni_sent in uni_sentence:
            bigram_sentence = u' '.join(bigram_model[uni_sent])
            f.write(bigram_sentence + '\n')

#Apply 'LineSentence' to bigrm_sentences_fp to break each review into individual
#sentences.
bigram_sentences = LineSentence(bigrm_sentences_fp)


trigram_model_pth = os.path.join(directory,'trigram_model_all.txt')

if 0 == 1:
    
#Make this if statement true (0 == 0) if you want to run code.
    trigram_model = Phrases(bigram_sentences)
    trigram_model.save(trigram_model_pth)
    

trigram_open = open(os.path.join(directory,"trigram.pickle"),"rb")
trigram_model = pickle.load(trigram_open)
trigram_open.close()

trigram_sentences_pth = os.path.join(directory,'trigram_sentences_all.txt')

if 0 == 1:
    
#Make this if statement true (0 == 0) if you want to run code
    with codecs.open(trigram_sentences_pth, 'w',encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence +'\n')

#Apply 'LineSentence' to trigram_sentences_pth to break each review into 
#individual sentences.
trigram_sentences = LineSentence(trigram_sentences_pth)

#Define function that merges three previously created dataframes with the 
#original training data. Shows the difference in the uni,bi and trigram words 
#compared to the original text
def normal_frame_check(uni,bi,tri,df):
    u,br,t = [],[],[]
    for a,b,c in zip(uni,bi,tri):
        u.append(u' '.join(a))
        br.append(u' '.join(b))
        t.append(u' '.join(c))
    mrg_df = pd.merge(pd.merge(pd.DataFrame(u),pd.DataFrame(br),
                      left_index=True,right_index=True),pd.DataFrame(t),
                        left_index=True,right_index=True)
    
    mrg_df.columns = ['Unigram_Sent','Bigram_Sent','Trigram_Sent']
    final_df = pd.merge(df,mrg_df,left_index=True,right_index=True)
    return final_df
frame_comp = normal_frame_check(uni_sentence,
                                bigram_sentences,
                                trigram_sentences,
                                train_new)

#Defined function to identify "stop words"
def stop_words(token):
    return token.is_stop

#Applying normalization to entire review as opposed to each sentence
trigram_reviews_path = os.path.join(directory,'trigram_reviews.txt')

#Define a function that applies normalization technique used before on the 
#entire review
def review_data(df,col):
    if 0 == 1:
        with codecs.open(trigram_reviews_path,'w',encoding='utf_8')as f:
            for a in df[col]:
                parsed=nlp(a)
                uni_review = [token.lemma_ for token in parsed
                      if not punct_space_removal(token)]
        
                bi_review = bigram_model[uni_review]
                tri_review = trigram_model[bi_review]
                tri_review = [t for t in tri_review
                              if t not in set(('and','or','not','but','to'))]
        
                tri_review = u' '.join(tri_review)
                f.write(tri_review + '\n')
                
print('part 1 complete')