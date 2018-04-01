#!/bin/bash
kubectl create -f ./manifests/sock-shop-ns.yaml
for filename in ./manifests/*; do
    echo $filename
#    /Users/jseverin/Documents/Microservices/microservices-demo/istio-0.2.12/bin/istioctl kube-inject -f filename | 
#    sed -e 's,istio/proxy_init:0.2.12,istio/proxy_init:0.2.12,' |
#    kubectl create -f -
    kubectl apply -f <(./istio-0.6.0/bin/istioctl kube-inject -f $filename)
done
