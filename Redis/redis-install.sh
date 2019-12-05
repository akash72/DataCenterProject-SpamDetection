#!/bin/sh
#
# This is the script you need to provide to install redis and start it running.
# It will be provided to the instance using redis-launch.sh
#
sudo apt-get update
sudo apt-get install -y python3 python3-pip git
sudo apt-get -y install redis-server
ps -f -u redis
sudo echo "bind 0.0.0.0 ::1" >> /etc/redis/redis.conf
sudo systemctl restart redis-server
ps -f -u redis