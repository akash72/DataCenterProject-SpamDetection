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
print("Enter Subject")
sub = input()
print("Enter Content")
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
print("Request Handled")

key = sub.split(' ', 1)[0]
print("KEY WORD",key)
    
time.sleep(30)
print("The result is ")
SendResult()

