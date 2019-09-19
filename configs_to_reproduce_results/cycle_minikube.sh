#!/usr/bin/env bash
minikube stop
minikube delete
minikube start --vm-driver=virtualbox --cpus=12 --memory=32000 --disk-size 65g

minikube addons enable heapster
minikube addons enable metrics-server
$(minikube docker-env)
sudo docker pull nicolaka/netshoot
