sudo apt update
#sudo apt install snapd
#sudo snap install kubectl --classic
apt-get update && apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
# start one command
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF
# end one command
apt-get update
apt-get install -y kubectl
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.27.0/minikube-linux-amd64 && chmod +x minikube && sudo mv minikube /usr/local/bin/
# need to install virtual box b/c dependencies :/ (from https://askubuntu.com/questions/779095/install-virtualbox-on-ubuntu-16-04-lts)
#Add PPA to sources.list
sudo nano /etc/apt/sources.list
#Add this file to the end of the file and do Ctrl+X to exit nano
deb http://download.virtualbox.org/virtualbox/debian xenial contrib
#Fetch the GPG key
wget -q https://www.virtualbox.org/download/oracle_vbox_2016.asc -O- | sudo apt-key add -
#Install the package
sudo apt update 
sudo apt install virtualbox-5.1
