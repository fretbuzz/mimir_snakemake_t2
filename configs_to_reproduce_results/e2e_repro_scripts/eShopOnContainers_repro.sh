#!/usr/bin/env bash
### eShopOnContainers REPRODUCABILITY -- SCALE ###
## TODO: parts of this have been tested... but it has NOT been tested e2e...

# setup minikube
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh

# install helm
curl -L https://git.io/get_helm.sh | bash
helm init --service-account tiller

# git clone this repo….
git clone https://github.com/dotnet-architecture/eShopOnContainers.git

# Go to the k8s folder in your local copy of the eShopOnContainers repo
cd ./eShopOnContainers
cd ./k8s/

############### need powershell
# Download the Microsoft repository GPG keys
wget -q https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb

# Register the Microsoft repository GPG keys
sudo dpkg -i packages-microsoft-prod.deb

# Update the list of products
sudo apt-get update

# Install PowerShell
sudo apt-get install -y powershell
#######################

sudo kubectl apply -f helm-rbac.yaml
sudo kubectl apply -f nginx-ingress/mandatory.yaml
sudo kubectl apply -f nginx-ingress/cm.yaml
sudo kubectl apply -f nginx-ingress/cloud-generic.yaml

cd ./helm
sudo pwsh ./deploy-all.ps1 -imageTag dev -useLocalk8s $true
sudo bash ./deploy-all.sh -t dev --use-local-k8s
# ^^ this step takes ~20 minutes...

## TODO: add in autoscaling...

## TODO: run experiments here...