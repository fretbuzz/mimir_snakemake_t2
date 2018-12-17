#!/usr/bin/env bash

echo "${1}"
echo "${2}"

kubectl get po --all-namespaces -o wide > "${1}"
kubectl describe node > "${2}"
