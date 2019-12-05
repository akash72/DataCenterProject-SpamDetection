#!/bin/sh
#
# This is the script you need to provide to install rabbitmq and start it running.
# It will be provided to the instance using rabbitmq-launch.sh
#
#!/bin/sh
#
# This is the script you need to provide to install rabbitmq and start it running.
# It will be provided to the instance using rabbitmq-launch.sh
#
#
# Erlang
#
export DEBIAN_FRONTEND=NONINTERACTIVE
wget -O- https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc | sudo apt-key add -
echo "deb https://packages.erlang-solutions.com/ubuntu bionic contrib" | sudo tee /etc/apt/sources.list.d/rabbitmq.list
apt update
apt -y install erlang
#
# Rabbitmq
#
wget -O- https://dl.bintray.com/rabbitmq/Keys/rabbitmq-release-signing-key.asc | sudo apt-key add -
wget -O- https://www.rabbitmq.com/rabbitmq-release-signing-key.asc | sudo apt-key add -
echo "deb https://dl.bintray.com/rabbitmq/debian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/rabbitmq.list
sudo apt update
sudo apt -y install rabbitmq-server
sudo echo "loopback_users=none" > /etc/rabbitmq/rabbitmq.conf
sudo systemctl restart rabbitmq-server