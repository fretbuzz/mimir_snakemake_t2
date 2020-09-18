# Install pre-requisites:
sudo apt-get update
sudo apt-get install -y python3-pip
git clone https://github.com/kubernetes-sigs/kubespray.git
cd ./kubespray/
sudo pip3 install -r requirements.txt

# Configure the kubespray params
cp -r inventory/sample inventory/mycluster
ips=$(cat /etc/hosts | tail -n 3| awk '{print $1}' | tr '\n' ' ')
declare -a IPS=("$ips")
CONFIG_FILE=inventory/mycluster/hosts.yml python3 contrib/inventory_builder/inventory.py ${IPS[@]}

# Setup passwordless-SSH:
cd ~/.ssh/
read  -n 1 -p "Setup the keys now (see doc for details)" mainmenuinput # TODO: automate this at some point.