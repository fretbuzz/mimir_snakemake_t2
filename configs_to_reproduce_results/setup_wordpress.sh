## WORK IN PROGRESS ##
#!/usr/bin/env bash
bash install_scripts/install_selenium_dependencies.sh
docker pull nicolaka/netshoot
wget https://get.helm.sh/helm-v2.14.1-linux-amd64.tar.gz
tar -zxvf helm-v2.14.1-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm
sudo python wordpress_setup/scale_wordpress.py 7
MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used

## TODO: there is more todo here! we nned to make sure that selenium works and that the rest goes through...