import argparse
import subprocess

from kubernetes_setup_functions import *


def main():
    wait_until_pods_done("kube-system")
    wait_until_pods_done("default")
    out = subprocess.check_output(["kubectl", "scale", "deploy", "orders",  "queue-master", "shipping",
                                    "--replicas=3", "--namespace=\"sock-shop\""])
    print out
    wait_until_pods_done("default") # wait until DB cluster is setup
    out = subprocess.check_output(["kubectl", "scale", "deploy", "catalogue", "front-end", "payment", "user",
                                   "--replicas=6", "--namespace=\"sock-shop\""])
    print out
    wait_until_pods_done("default") # wait until DB cluster is setup
    out = subprocess.check_output(["kubectl", "scale", "deploy", "cart", "--replicas=4", "--namespace=\"sock-shop\""])
    print out
    wait_until_pods_done("default") # wait until DB cluster is setup

if __name__== "__main__":
    main()