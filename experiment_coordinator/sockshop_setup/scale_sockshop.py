import argparse
import subprocess
import time
from kubernetes_setup_functions import *
import os


def main(deployment_scaling, autoscale_p):
    deploy_sockshop(deployment_scaling, autoscale_p)
    scale_sockshop(deployment_scaling, autoscale_p)

def deploy_sockshop(deployment_scaling, autoscale_p):

    wait_until_pods_done("kube-system")
    time.sleep(60)
    try:
        out = subprocess.check_output(["kubectl", "create", "-f", "./sockshop_setup/sock-shop-ns.yaml", "-f",
                                   "./sockshop_setup/sockshop_modified.yaml"])
    except:
        pass
    time.sleep(60)
    wait_until_pods_done("sock-shop")

def scale_sockshop(deployment_scaling, autoscale_p):
    orders_containers = deployment_scaling['orders']['max']
    queue_master_containers = deployment_scaling['queue-master']['max']
    shipping_containers = deployment_scaling['shipping']['max']
    out1 = subprocess.check_output(["kubectl", "scale", "deploy", "orders", "--replicas=" + str(orders_containers), "--namespace=sock-shop"])
    out2 = subprocess.check_output(["kubectl", "scale", "deploy", "queue-master", "--replicas=" + str(queue_master_containers), "--namespace=sock-shop"])
    out3 = subprocess.check_output(["kubectl", "scale", "deploy", "shipping", "--replicas=" + str(shipping_containers), "--namespace=sock-shop"])
    print out1, out2, out3
    wait_until_pods_done("sock-shop")
    catalogue_containers = deployment_scaling['catalogue']['max']
    front_end_containers = deployment_scaling['front-end']['max']
    payment_containers = deployment_scaling['payment']['max']
    user_containers = deployment_scaling['user']['max']
    out1 = subprocess.check_output(["kubectl", "scale", "deploy", "catalogue", "--replicas=" + str(catalogue_containers), "--namespace=sock-shop"])
    out2 = subprocess.check_output(["kubectl", "scale", "deploy", "front-end", "--replicas=" + str(front_end_containers), "--namespace=sock-shop"])
    out3 = subprocess.check_output(["kubectl", "scale", "deploy", "payment",  "--replicas=" + str(payment_containers), "--namespace=sock-shop"])
    out4 = subprocess.check_output(["kubectl", "scale", "deploy",  "user", "--replicas=" + str(user_containers), "--namespace=sock-shop"])
    print out1,out2,out3,out4
    wait_until_pods_done("sock-shop")
    cart_containers = deployment_scaling['cart']['max']
    out = subprocess.check_output(["kubectl", "scale", "deploy", "cart", "--replicas=" + str(cart_containers), "--namespace=sock-shop"])
    print out
    wait_until_pods_done("sock-shop")
    time.sleep(240)

    if autoscale_p:
        out = subprocess.check_output(["minikube", "addons", "enable", "heapster"])
        print out
        out = subprocess.check_output(["minikube", "addons", "enable", "metrics-server"])
        print out
        try:
            out = subprocess.check_output(["git", "clone", "https://github.com/microservices-demo/microservices-demo.git"])
            print out
        except:
            pass
        out = subprocess.check_output(["kubectl", "apply", "-f", "./microservices-demo/deploy/kubernetes/autoscaling/"])
        print out

        # NOTE the front-end microservice crashes fairly regularly... it's important that there's a couple more so one's always live
        autoscale_cmd_str = ["kubectl", "autoscale", "deployment", "front-end", "--min=" + str(3), "--max=" + str(
            10),"--cpu-percent=" + str(50), "--namespace=sock-shop"]
        try:
            out = subprocess.check_output(autoscale_cmd_str)
            print out
        except:
            pass

if __name__== "__main__":
    os.chdir("..")
    deployment_scaling = {}
    deployment_scaling['orders'] = {"min": 3, "max": 3}
    deployment_scaling['queue-master'] = {"min": 3, "max": 3}
    deployment_scaling['shipping'] = {"min": 3, "max": 3}
    deployment_scaling['catalogue'] = {"min": 6, "max": 6}
    deployment_scaling['front-end'] = {"min": 6, "max": 6}
    deployment_scaling['payment'] = {"min": 6, "max": 6}
    deployment_scaling['user'] = {"min": 6, "max": 6}
    deployment_scaling['cart'] = {"min": 4, "max": 4}

    main(deployment_scaling, False)