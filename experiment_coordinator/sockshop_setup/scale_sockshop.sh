#!/usr/bin/env bash

declare -i current_carts_instances=3
declare -i goal_carts_instances=3

kubectl scale deploy orders  queue-master shipping --replicas=3 --namespace="sock-shop"
sleep 40
kubectl scale deploy catalogue front-end payment user --replicas=6 --namespace="sock-shop"
sleep 40
kubectl scale deploy cart --replicas=4 --namespace="sock-shop"
sleep 40
#sleep 360

#while [ $current_carts_instances -lt $goal_carts_instances ]
#do
#current_carts_instances=$current_carts_instances+1
#kubectl scale deploy cart "--replicas=$current_carts_instances" --namespace="sock-shop"

#sleep 240
#done

