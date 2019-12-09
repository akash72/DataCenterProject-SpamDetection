Final Project for Data Center Scale Computing 

RabbitMQ Instance

>Open the Gcloud command line interface and run the rabitmq_instance.py to create a vm for rabbitmq under the name rabbitmq.


Redis 

> In the Gcloud command line interface run the redis_instance.py to create a vm for redis under the name redis.

REST

>In the Gcloud command line interface run the rest_instance.py to create a vm for rest under the name rest.

Worker
> In GCP create a single node virtual machine under the name worker.

Working

> Once all the four instances are up and running. Upload the server.py and client.py files on the rest instance.
> Upload the worker.py, install.sh and requirements.txt on the worker instance. Run the install.sh file on the worker node to install dependencies.
> First run the server.py file.
> Once server is up and running run the worker.py file in the worker node.
> Once both server.py and worker.py files are both up and running run the client.py file to send the request body (Subject and email body)