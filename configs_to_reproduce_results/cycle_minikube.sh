#!/usr/bin/env bash
sudo minikube stop
sudo minikube delete
sudo minikube start --vm-driver=virtualbox --cpus=12 --memory=32000 --disk-size 65g

sudo minikube addons enable heapster
sudo minikube addons enable metrics-server
eval $(sudo minikube docker-env)
