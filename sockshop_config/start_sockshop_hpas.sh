#!/bin/bash
for filename in ./microservices-demo/deploy/kubernetes/autoscaling/*; do
    echo $filename
    kubectl apply -f $filename
done
