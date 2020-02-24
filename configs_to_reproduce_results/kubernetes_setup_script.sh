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
# ren-enable the folllowing if my new method does not work...
#wget -q https://www.virtualbox.org/download/oracle_vbox_2016.asc -O- | sudo apt-key add -
#wget -q https://www.virtualbox.org/download/oracle_vbox.asc -O- | sudo apt-key add -
#sudo sh -c 'echo "deb http://download.virtualbox.org/virtualbox/debian $(lsb_release -sc) contrib" >> /etc/apt/sources.list.d/virtualbox.list'
#sudo apt update
#sudo apt-get install virtualbox-6.0 -y
#### vboxmanage setproperty machinefolder /mydata/
# try this...
wget https://download.virtualbox.org/virtualbox/6.1.4/virtualbox-6.1_6.1.4-136177~Ubuntu~xenial_amd64.deb
sudo dpkg -i virtualbox-6.1_6.1.4-136177~Ubuntu~xenial_amd64.deb
sudo apt-get install libcurl3 libopus0 libsdl1.2debian
sudo apt-get -f install


# okay, onto the next step
# First, we'll install and start Minikube
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v1.3.1/minikube-linux-amd64  && chmod +x minikube
sudo cp minikube /usr/local/bin && rm minikube

sudo minikube start --vm-driver=virtualbox --cpus=12 --memory=32000 --disk-size 65g

# and make sure to enable certain addons
sudo minikube addons enable heapster
sudo minikube addons enable metrics-server

# Second, we'll install the experimental coordinator's dependencies.

### setup docker (need to use a specific version because one of the newer ones was causing problems)
#curl -fsSL https://get.docker.com -o get-docker.sh
#sudo sh get-docker.sh
#sudo apt-cache policy docker-ce #???
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable"
sudo apt-get update
sudo apt-get -y install docker-ce=17.06.0~ce-0~ubuntu

eval $(sudo minikube docker-env)
sudo docker pull nicolaka/netshoot

### then pip to handle the python dependencies
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py

### then use pip to handle the python dependencies
sudo apt-get -y install python-dev
sudo pip install requests pyasn1 ipaddress urllib3 --upgrade
sudo pip install docker pexpect netifaces selenium kubernetes requests
sudo pip install locustio==0.13.5 pycryptodome
sudo pip install pwntools
sudo python -m easy_install --upgrade pyOpenSSL

# docker-machine can be useful for certificate management, so installing that is a good idea
### following the official instructions: https://docs.docker.com/machine/install-machine/
base=https://github.com/docker/machine/releases/download/v0.16.0 &&
  curl -L $base/docker-machine-$(uname -s)-$(uname -m) >/tmp/docker-machine &&
  sudo install /tmp/docker-machine /usr/local/bin/docker-machine

# Fourth let's use the experimental coordinator to deploy the Sockshop application and start an experiment.
git clone https://github.com/fretbuzz/mimir_v2.git
cd ../
