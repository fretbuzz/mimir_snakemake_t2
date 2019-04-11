#!/usr/bin/env bash

# note: might wanna do something fancier in the future... for now it'll be very boring

kubectl autoscale deployment adservice --cpu-percent=80 --min=1 --max=10
####
kubectl autoscale deployment cartservice --cpu-percent=80 --min=1 --max=10 # doesn't distribute traffic weell
kubectl autoscale deployment checkoutservice --cpu-percent=80 --min=1 --max=1 # replicas aren't used... keep to 1.
#####
kubectl autoscale deployment currencyservice --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment emailservice --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment frontend --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment loadgenerator --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment paymentservice --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment productcatalogservice --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment recommendationservice --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment redis-cart --cpu-percent=80 --min=1 --max=10
kubectl autoscale deployment shippingservice --cpu-percent=80 --min=1 --max=10