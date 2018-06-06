#!/bin/bash
kubectl create -f ./microservices-demo/deploy/kubernetes/manifests/sock-shop-ns.yaml
for filename in ./microservices-demo/deploy/kubernetes/manifests/*; do
    echo $filename
    kubectl apply -f <($1/bin/istioctl kube-inject -f $filename)
done
