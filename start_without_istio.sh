#!/bin/bash
kubectl create -f ./manifests/sock-shop-ns.yaml
for filename in ./manifests/*; do
    echo $filename
    kubectl apply -f $filename
done
