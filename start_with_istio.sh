#!/bin/bash
kubectl create -f ./manifests/sock-shop-ns.yaml
for filename in ./manifests/*; do
    echo $filename
    kubectl apply -f <($1/bin/istioctl kube-inject -f $filename)
done
