#!/usr/bin/env bash

echo "${1}"
echo "${2}"
echo "${3}"
echo "${4}"
echo "${5}"

if [ "$5" == "--clear_files" ]; then
  echo "" > "${1}"
  echo "" > "${2}"
  echo "" > "${3}"
  echo "" > "${4}"
  exit 0
fi

kubectl get po --all-namespaces -o wide >> "${1}"
echo -e "\n=======\n" >> "${1}"

kubectl describe node >> "${2}"
echo -e "\n=======\n" >> "${2}"

kubectl get deploy -o wide --all-namespaces  >> "${3}"
echo -e "\n=======\n" >> "${3}"

eval $(sudo minikube docker-env)
docker ps -a >> "${4}"
echo -e "\n=======\n" >> "${4}"