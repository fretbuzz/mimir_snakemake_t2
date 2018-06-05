#!/bin/bash
kubectl create -f ./microservices-demo/munnin/microservices-demo/deploy/kubernetes/manifests/sock-shop-ns.yaml
for filename in ./microservices-demo/munnin/microservices-demo/deploy/kubernetes/manifests/*; do
    echo $filename
    kubectl apply -f $filename
done
