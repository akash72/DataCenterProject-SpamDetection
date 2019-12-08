#!/usr/bin/env python

# importing the required libraries
import requests
import sys
import pika
import json
import pickle
import time

# storing the values for future use
addr = 'http://{}:5000'
print("Enter the subject of your mail :")
sub = input()
print("Enter the content of your mail :")
con = input()
headers = {'content-type': 'text/plain'}

def detectSpam():
    spam_url = addr + '/api/spam-check/' + sub + '/' + con
    # generate request for image
    response = requests.post(spam_url, headers=headers)
    print(response.text)


def SendResult():
    spam_res = addr + '/api/key/' + key
    response = requests.post(spam_res, headers=headers)
    print(response.text)

addr = addr.format(sys.argv[1])

detectSpam()
print("#"*50)
print("......Your mail is being processed......")

key = sub.split(' ', 1)[0]
print("KEY WORD",key)
    
time.sleep(30)
print("......Your mail is processed......")
print("The result is : ")
SendResult()

