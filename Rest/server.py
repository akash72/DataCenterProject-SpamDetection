from flask import Flask, request, Response
import jsonpickle
import io
import hashlib
import sys
import pika
import json
import pickle
import redis
import base64

# Initialize the Flask application
app = Flask(__name__)

credentials = pika.PlainCredentials('guest', 'guest')
parameters = pika.ConnectionParameters('10.128.15.216', 5672, '/', credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.queue_declare(queue='work')

# route http posts to this method
@app.route('/api/spam-check/<X>/<Y>', methods=['POST'])
def test(X, Y):
    sub = X
    con = Y
    print(sub)
    print(con)
    text = sub + " " + con
    r = request
    first_word = text.split(' ', 1)[0]
    # convert the data to a PIL image type so we can extract dimensions
    try:
        m = hashlib.md5(first_word.encode())
        q = m.hexdigest()
        response = {
            "mail": q
        }
        response_pickled = jsonpickle.encode(response)
        print("HASH OF KEY")
        print(q)
        message = pickle.dumps([text, q, r.data])
        channel.basic_publish(exchange='', routing_key='work', body=message)
        print(" [x] Sent Data ")
        connection.close()
    except:
        response = {
           "mail": None
        }

    return Response(response=response_pickled, status=200, mimetype="application/json")

#Get Result of keyword
@app.route('/api/key/<X>', methods=['POST'])
def getvalue(X):
    r = redis.Redis(host="10.128.15.215", db=1)
    v = r.get(X)
    r2 = redis.Redis(host="10.128.15.215", db=2)
    #result = base64.b64decode(pickle.loads(r2.get(v)))
    result = pickle.loads(r2.get(v))
    response = {
        "The Results is": result
    }
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")
    
app.run(host="0.0.0.0", port=5000)
app.debug = True



