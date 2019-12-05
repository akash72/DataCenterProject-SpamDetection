#!/usr/bin/env python

# importing the required libraries
import requests
import sys
import pika
import json
import pickle

# storing the values for future use
addr = 'http://{}:5000'
print("Enter Subject")
sub = input()
print("Enter Content")
con = input()
headers = {'content-type': 'text/plain'}

credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('10.128.15.216', 5672, '/', credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
#channel.queue_declare(queue='work')
channel.queue_declare(queue='proj')

def detectSpam():
    spam_url = addr + '/api/spam-check/' + sub + '/' + con
    # generate request for image
    response = requests.post(spam_url, headers=headers)
    print(response.text)

def callback(ch, method, properties, body):
    obj1 = body
    k = obj1
    print(k)

addr = addr.format(sys.argv[1])

detectSpam()
print("Request Handled")
channel.basic_consume(
    queue='proj', on_message_callback=callback, auto_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
