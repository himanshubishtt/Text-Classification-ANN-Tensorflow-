#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the necessary modules

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[3]:


#reading the dataset from the csv file provided

data = pd.read_csv(r"C:\Users\Home\Downloads\bbc-text.csv",delimiter = ",")
data.head(5)


# In[4]:


# Observing and understanding the dataset

print(data.shape)
print(data.category.value_counts())


# In[5]:


# Performing the traintest split.
traintext,testtext,trainlabel,testlabel = train_test_split(data.text,data.category,test_size=0.3)


# In[6]:


# Now we will clean the textual data


# In[7]:


# Making a list of stopwords to be removed including useless punctuations

stop=stopwords.words('english')+list(punctuation)


# In[8]:


# This function checks each word and adds it to a list if it is not a stopword

def clean(words):
    output=[]
    for w in words:
        if w.lower() not in stop:
            output.append(w)
    return output


# In[9]:


# Removing stopwords from the textual data

documents=[]
for a,b in zip(traintext,trainlabel):
    documents.append((clean(word_tokenize(a)),b))


# In[10]:


# This function clubs similar kinds of words together and converts them to their root word. This is called Lemmatization.

def wordlemmatizer(words):
    output=[]
    lemmatizer=WordNetLemmatizer()
    for w in words:
        pos=pos_tag([w])
        cleanword=lemmatizer.lemmatize(w,pos=simplepos(pos[0][1]))
        output.append(cleanword.lower())
    return output


# Since POS tags and Lemmatizer use different word tags, this function converts a pos tag to a simpler lemmatizer tag.

def simplepos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
        
    else:
        return wordnet.NOUN


# In[11]:


# Now we lemmatize the text documents to process them easily 

new=[(wordlemmatizer(document),category) for document,category in documents]


# In[12]:


# After cleaning the dataset, we keep labels and textual data seperately in two lists for further processing on each

categories=[category for document,category in new]
textdocs=[" ".join(document) for document,category in documents]


# In[13]:


# The Count Vectorizer takes the textual data and uses n most common words as features.
# It then creates a sparse matrix showing the count of occurence of n most common words in each document
# This is also called a Bag of Words model of textual data

count_vec = CountVectorizer(max_features=100)
xtrainfeaturesparse = count_vec.fit_transform(textdocs)


# In[14]:


# This is done because the neural network can't accept the textual data directly and we need data in X and Y format that classifiers are familiar with

xtrainfeatures = xtrainfeaturesparse.todense()
print(xtrainfeatures)


# In[15]:


# One Hot encoding is done to feed the labels into the neural network

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(categories)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_trainclass = onehot_encoder.fit_transform(integer_encoded)
print(onehot_trainclass)


# In[16]:


# In the final predictions, these numbers will represent the category of the text given

print(label_encoder.classes_)


# In[17]:


# Classification is done using an ANN (Artificial Neural Network)
# For that I import the library Tensor Flow

import tensorflow as tf


# In[18]:


# I use 2 hidden layers having 1024 units each

n_input = 100
n_hidden_1 = 1024
n_hidden_2 = 1024
n_classes = 5

# Initializing the weights and biases of the network with random values initially which will get optimized later

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[19]:


# This function does the forward propagation in the neural network and returns the output

def forward_propagation(x, weights, biases):
    in_layer1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    out_layer1 = tf.nn.relu(in_layer1) # The Activation function is relu 
    
    in_layer2 = tf.add(tf.matmul(out_layer1, weights['h2']), biases['h2'])
    out_layer2 = tf.nn.relu(in_layer2)
    
    output = tf.add(tf.matmul(out_layer2, weights['out']), biases['out'])
    return output


# In[191]:


# Here, I create two placeholder which will be initialized later as the input data and input labels
 
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder(tf.int32, [None, n_classes])
pred = forward_propagation(x, weights, biases)


# In[214]:


# The cost fuction I used was the softmax cross entropy function
# It measures the probability error in discrete classification tasks in which the classes are mutually exclusive

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))


# In[270]:


# The optimizer I used was the AdamOptimizer with a learning rate of 0.01
# Adam is an adaptive learning rate method, which means, it computes individual learning rates for different parameters

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimize = optimizer.minimize(cost)


# In[271]:


#Initializing the variables in tensorflow

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[272]:


# These two are plotted to observe the loss changes during the iterations

cost_array = []
epoch_array = []


# In[273]:


# Fitting the training data to the data model by minimizing cost and optimizing weights 500 times

for i in range(500):
    c, _ = sess.run([cost,optimize], feed_dict={x:xtrainfeatures, y: onehot_trainclass})
    cost_array.append(c)
    epoch_array.append(i)
    print(i," ",c)


# In[27]:


# Processing the testing data to evaluate predictions. The processing is similar to what I've done to the training data

testdocs=[]
for i in testtext:
    testdocs.append(clean(word_tokenize(i)))


# In[28]:


testdocs=[wordlemmatizer(doc) for doc in testdocs]


# In[29]:


testdocs=[" ".join(document) for document in testdocs]


# In[30]:


xtestfeaturesparse=count_vec.transform(testdocs)


# In[31]:


xtestfeatures = xtestfeaturesparse.todense()


# In[32]:


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(testlabel)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_testclass = onehot_encoder.fit_transform(integer_encoded)
print(onehot_testclass)


# In[274]:


# Predicting the labels of the testing data.

predictions = tf.argmax(pred, 1)
correct_labels = tf.argmax(y, 1)
correct_predictions = tf.equal(predictions, correct_labels)
predictions,correct_predictions  = sess.run([predictions, correct_predictions], feed_dict={x:xtestfeatures,
                                              y:onehot_testclass})
correct_predictions.sum()


# In[275]:


# I got an accuracy of 86.97% in the testing data

accuracy = correct_predictions.sum()/len(correct_predictions)
print(accuracy)


# In[185]:


# In predictions, 0 : Business 1 : Entertainment 2 : Politics 3 : Sport 4 : Tech
print(label_encoder.classes_)


# In[276]:


predictions


# In[277]:


# Decoding the predictions back to normal

submission = []
for i in predictions:
    if i == 0:
        submission.append('business')
    elif i == 1:
        submission.append('entertainment')
    elif i == 2:
        submission.append('politics')
    elif i == 3:
        submission.append('sport')
    elif i == 4:
        submission.append('tech')


# In[278]:


submission=pd.DataFrame(submission)
submission.head(5)


# In[279]:


# Saving predictions to a csv file

submission.to_csv(r"C:\Users\Home\Desktop\predictions.csv")


# In[280]:


# Now to plot the epoch vs loss graph for test data we reinitialize the optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimize = optimizer.minimize(cost)


# In[281]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[282]:


test_cost_array = []
test_epoch_array = []
for i in range(500):
    c, _ = sess.run([cost,optimize], feed_dict={x:xtestfeatures, y: onehot_testclass})
    test_cost_array.append(c)
    test_epoch_array.append(i)
    print(i," ",c)


# In[297]:


# Plotting the epochs vs loss graph

plt.plot(test_epoch_array,test_cost_array,color="cyan",label="Test")
plt.plot(epocpochh_array,cost_array,color="orange",label="Train")
plt.xlabel("Epochs/Iterations")
plt.ylabel("Cost/Loss")
plt.title('Epochs vs Loss Graph')
plt.axis([0,100,0,7000])
plt.legend()
plt.show

