#!/usr/bin/env python
from flask import request
import pika
import redis
import sys
import json
import pickle
import jsonpickle
import os
import glob
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split


def read(file_path, em, la, is_spam):
    for files in glob.glob(os.path.join(file_path, "*txt")):
        with open(files, "r", encoding='ISO-8859-1') as fp:
            em.append(fp.read())
            la.append(is_spam)

    return em, la


def read_mails(file_path):
    em = []
    la = []

    em, la = read(file_path[0], em, la, 0)
    em, la = read(file_path[1], em, la, 1)
    em = clean_text(em)
    return em, la


def letter_only(string):
    return string.isalpha()


def clean_text(docs):
    all_names = set(names.words())
    cl_doc = []
    lemm = WordNetLemmatizer()
    for doc in docs:
        cl_doc.append(" ".join([lemm.lemmatize(word.lower()) for word in doc.split() if letter_only(word) and word not in all_names]))

    return cl_doc


def get_label_index(la):
    ''' Label the index
        label is of the list
        return dict object 
    '''
    from collections import defaultdict
    lab_ind = defaultdict(list)
    for ind,lab in enumerate(la):
        lab_ind[lab].append(ind)

    return lab_ind


def get_prior(lab_ind):
    '''Compute the prior knowlege based on trainning set
        i.e compute the probabitlity of legit and spame emil(0,1)
        format of label_index
        [{0:index}]
    '''
    prior = {lab:len(ind) for lab,ind in lab_ind.items()}
    total_count = sum(prior.values())
    for lab in prior:
        prior[lab] = float(prior[lab]/total_count)

    return prior


def get_likelihood(email_matrix,label_index,smoothing = 0):
    '''
    Computing likelihood based on trainning set
    sum(axis = 0) = computing total word occrued in all text of that particular frequency
    Return 
    dictonary like object
    '''
    likelihood = {}
    for label,index in label_index.items():
        likelihood[label] = email_matrix[index,:].sum(axis= 0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0] #2D -> 1D
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label]/float(total_count)

    return likelihood 


def get_posterior(email_matrix,prior,likelihood):
    '''Computing posterior smaple based on prior and likelihood '''
    num_email = (email_matrix.shape[0])
    posteriors = []
    for i in range(num_email):
        posterior = {key:np.log(prior_label) for  key, prior_label in prior.items()}
        for label,likelihood_label in likelihood.items():
            #get email(i)
            email_vector = email_matrix.getrow(i)
            word_count = email_vector.data
            word_indices = email_vector.indices
            for count,index in zip(word_count,word_indices):
                # likelihood[index] -> average freq from trainning set * count
                posterior[label] += np.log(likelihood_label[index]) * count
        min_log_posterior = min(posterior.values())

        #Converting back to original value
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                posterior[label] = float('inf')
        sum_posterior = sum(posterior.values())

        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())

    return posteriors


credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('10.128.15.216', 5672, '/', credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.queue_declare(queue='work')


def callback(ch, method, properties, body):
    print("Callback function is called!")
    obj1, obj2, obj3 = pickle.loads(body)
    text = obj1
    hashval = obj2
    first_Word = text.split(' ', 1)[0]
    #print("First Word",first_Word)
    print("#"*50)

    print("Processing the mail.....")
    file_path_legit = os.path.join(os.path.dirname(os.path.realpath(__file__)),"enron1/ham/")
    file_path_spam = os.path.join(os.path.dirname(os.path.realpath(__file__)),"enron1/spam/")

    emails,labels = read_mails([file_path_legit, file_path_spam])
    cv = CountVectorizer(stop_words = "english",max_features = 8000)

    Xtrain,Xtest,Ytrain,Ytest = train_test_split(emails,labels,test_size = 0.0001,random_state = 42)

#   Trainning data 
    email_train = cv.fit_transform(Xtrain)
    label_index = get_label_index(Ytrain)
    prior = get_prior(label_index)
    likelihood = get_likelihood(email_train,label_index,smoothing = 0.5)

# Testing data
    correct = 0.0
    Xt = []
    Xt.append(text)
    print(Xt)
    email_test = cv.transform(Xt)
    print(email_test)
    results = get_posterior(email_test,prior,likelihood)
    print(results)
    Yt = [0,1]
    k = None
    for result, actual in zip(results, Yt):
        if actual == 1 or actual == 0:
            if result[0] > 0.8 and result[1] < 0.3:
                correct += 1
                k = "Ham"
            elif result[1] > 0.5 and result[0] < 0.7:
                correct += 1
                k = "Spam"
    #if(k):
     #   print(k)
    #else:
     #   k = "SPAM"
      #  print(k)
    print("ACCURACY:{0:.1f}".format(correct/len(Yt) *100))

    response = {
            "res": k
        }
    print("The mail is a " + k)

    redisHash = redis.Redis(host="10.128.15.215 ", db=1)     # Keyword - Hash(content) pair
    redisHash.set(first_Word, hashval)
    print("\n")
    print("#"*50)
    print("\n Redis get by Key word of mail: ", first_Word)
    val1 = redisHash.get(first_Word)
    print(val1)

    listval = []
    listval.append(text)
    listval.append(k)
    redisName = redis.Redis(host="10.128.15.215 ", db=2)        # Store Hash - [Content,result] Pair
    redisName.set(hashval, pickle.dumps(listval))
    print("\n")
    print("#"*50)
    print("\n Redis get by Hash: ", hashval)
    val2 = pickle.loads(redisName.get(hashval))
    print(val2)
    print("Sent worker data")


channel.basic_consume(
        queue='work', on_message_callback=callback, auto_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

