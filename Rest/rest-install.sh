#!/bin/sh
#
# This is the script you need to provide to install the rest-server.py and start it running.
# It will be provided to the instance using redis-launch.sh
#
#!/bin/bash
#!/usr/bin/python


sudo apt-get update
sudo apt-get install -y python3 python3-pip git
sudo pip3 install Flask
sudo pip3 install pillow jsonpickle
pip3 install requests
pip3 install numpy
sudo pip3 install --upgrade pika
