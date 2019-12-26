#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Implementation of SentClass3 with 1lac training data- using Unidirectional LSTM . Gave 76% test accuracy

import numpy as np
from utils import *
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


# In[11]:


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')


# In[12]:


data=pd.read_csv('data/twitter_train_data_correct.csv')


# In[13]:


#0 is negative and 1 is positive
data.loc[data['target'] == 4, 'target'] = 1


# In[14]:


data['p_text'] = data['text']


# In[15]:


def remove_url(text):
    text = re.sub(re.compile(r'http\S+'), "",text)
    return text

def remove_mentions(text):
    text = re.sub(re.compile(r'@\S+'), "",text)
    return text

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

data['p_text'] = data['p_text'].apply(lambda x: remove_url(x))
data['p_text'] = data['p_text'].apply(lambda x: remove_mentions(x))
data['p_text'] = data['p_text'].apply(lambda x: remove_punct(x))

#Now 'p_text' has processed text


# In[16]:


#run only once to find maximum length of whole set
all_texts = np.array(data['p_text'])
maxLen = 0
for i in all_texts:
    Len = len(i.split())  #problem1
    if(Len > maxLen):
        maxLen = Len
maxLen += 5


# In[17]:


train_set = data.sample(n = 100000)
val_set = data.sample(n = 20000)
test_set = data.sample(n=10000)


# In[18]:


train_X = np.array(train_set["p_text"])
train_Y = np.array(train_set["target"])
val_X = np.array(val_set["p_text"])
val_Y = np.array(val_set["target"])
test_X = np.array(test_set["p_text"])
test_Y = np.array(test_set["target"])


# In[19]:


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                                   # number of training examples
    X_indices = np.zeros((m,max_len))
    all_keys = word_to_index.keys()
    for i in range(m):                               
        sentence_words = X[i].lower().split()
        j = maxLen-len(X[i].lower().split())
        for w in sentence_words:
            if w in all_keys:
                X_indices[i, j] = word_to_index[w]
            j = j+1
  
    return X_indices


# In[20]:


train_X_indices = sentences_to_indices(train_X, word_to_index, maxLen)
val_X_indices = sentences_to_indices(val_X, word_to_index, maxLen)
test_X_indices = sentences_to_indices(test_X, word_to_index, maxLen)


# In[21]:


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement) [unk token]
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]
        
    embedding_layer = Embedding(vocab_len,emb_dim, trainable=False)
    
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# In[22]:


def sentiSid(input_shape, word_to_vec_map, word_to_index):
    
    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = Input(shape=input_shape,dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences.
    X = LSTM(128,return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128,return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)
    
    # Propagate X through a Dense layer with 1 units
    X = Dense(1)(X)
    # Add a softmax activation
    X = Activation('sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    return model


# In[23]:


model = sentiSid((maxLen,), word_to_vec_map, word_to_index)
model.summary()


# In[24]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[25]:


model.fit(train_X_indices, train_Y,
          batch_size=32,
          epochs=50,
          validation_data=(val_X_indices, val_Y))


# In[26]:


loss, acc = model.evaluate(test_X_indices, test_Y)


# In[27]:


print(acc)


# In[28]:


model.save('SentClass7model.h5')


# In[29]:


SentClass7_json = model.to_json()


# In[30]:


with open("SentClass7json.json", "w") as json_file:
    json_file.write(SentClass7_json)


# In[31]:


model.save_weights('SentClass7weights.h5')


# In[ ]:




