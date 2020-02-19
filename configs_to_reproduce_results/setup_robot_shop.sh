#!/usr/bin/env bash
docker pull nicolaka/netshoot
wget https://get.helm.sh/helm-v2.14.1-linux-amd64.tar.gz
tar -zxvf helm-v2.14.1-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm

git clone https://github.com/instana/robot-shop.git
cd robot-shop/K8s/helm
helm install --name robot-shop --namespace robot-shop .