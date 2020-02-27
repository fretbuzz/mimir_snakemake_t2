#!/usr/bin/env bash
sudo docker pull nicolaka/netshoot
wget https://get.helm.sh/helm-v2.14.1-linux-amd64.tar.gz
tar -zxvf helm-v2.14.1-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm

git clone https://github.com/instana/robot-shop.git
cd robot-shop/K8s/helm
sudo helm init --wait
sudo helm install --name robot-shop --namespace robot-shop .
sed -i  's/max 2/max 4/' ../autoscale.sh # b/c we want max # of replicas to be 4, not 2...
sudo bash ../autoscale.sh
cd ../../../ # gotta get back to the experimental_coordinator directory...
