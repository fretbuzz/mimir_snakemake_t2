sudo apt-get update

# This video will setup the Minikube Kubernetes environment, deploy a microservice application, and then collect some network activity data while simulating user traffic.
# Zeroth, we'll install the Minikube dependencies.

# We'll start by installing Kubectl.
### from the official instructions: https://kubernetes.io/docs/tasks/tools/install-kubectl/#install-kubectl-on-linux
curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
kubectl version

# The we'll install the VirtualBox Driver:
# Following from here:: https://websiteforstudents.com/virtualbox-6-0-is-out-heres-how-to-install-upgrade-on-ubuntu-16-04-18-04-18-10/
sudo apt-get install gcc make linux-headers-$(uname -r) dkms -y
wget -q https://www.virtualbox.org/download/oracle_vbox_2016.asc -O- | sudo apt-key add -
wget -q https://www.virtualbox.org/download/oracle_vbox.asc -O- | sudo apt-key add -
sudo sh -c 'echo "deb http://download.virtualbox.org/virtualbox/debian $(lsb_release -sc) contrib" >> /etc/apt/sources.list.d/virtualbox.list'
sudo apt update
sudo apt-get install virtualbox-6.0 -y
#### vboxmanage setproperty machinefolder /mydata/

# okay, onto the next step
# First, we'll install and start Minikube
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64   && chmod +x minikube
sudo cp minikube /usr/local/bin && rm minikube

sudo minikube start --vm-driver=virtualbox --cpus=12 --memory=32000 --disk-size 65g

# and make sure to enable certain addons
sudo minikube addons enable heapster
sudo minikube addons enable metrics-server

# Second, we'll install the experimental coordinator's dependencies.

### setup docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

eval $(sudo minikube docker-env)
sudo docker pull nicolaka/netshoot

### then pip to handle the python dependencies
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py

### then use pip to handle the python dependencies
sudo pip install requests pyasn1 ipaddress urllib3 --upgrade
sudo pip install docker pexpect netifaces selenium kubernetes requests
sudo pip install locustio

# docker-machine can be useful for certificate management, so installing that is a good idea
### following the official instructions: https://docs.docker.com/machine/install-machine/
base=https://github.com/docker/machine/releases/download/v0.16.0 &&
  curl -L $base/docker-machine-$(uname -s)-$(uname -m) >/tmp/docker-machine &&
  sudo install /tmp/docker-machine /usr/local/bin/docker-machine

# Fourth let's use the experimental coordinator to deploy the Sockshop application and start an experiment.
git clone https://github.com/fretbuzz/mimir_v2.git
cd ../
