#!/bin/bash

sudo apt-get update
sudo apt-get -y install python3 python3-pip 
sudo pip3 install --upgrade pika
sudo pip3 install pillow jsonpickle
pip3 install requests
pip3 install redis

pip3 install -r ./requirements.txt
