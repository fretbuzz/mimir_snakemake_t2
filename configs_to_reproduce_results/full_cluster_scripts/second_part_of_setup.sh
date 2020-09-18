sudo chmod 600 tempkey.pem  
sudo chmod 600 tempkey.pub 
eval `ssh-agent`
ssh-add ~/.ssh/tempkey.pem # TODO: automate using your password here

# Run the kubespray ansible playbook
cd ~/kubespray
ansible-playbook -i inventory/mycluster/hosts.yml  --become --become-user=root cluster.yml

# Copy kubeconfig to the home directory
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# install istio
curl -L https://istio.io/downloadIstio | sh - # TODO: version control this to 1.7.0
cd istio-1.7.1 
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo
kubectl label namespace default istio-injection=enabled
kubectl create namespace sock-shop
kubectl label namespace sock-shop istio-injection=enabled # TODO: handle other applications than just sockshop

kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.7/samples/addons/prometheus.yaml

# Install InfluxDB
wget -qO- https://repos.influxdata.com/influxdb.key | sudo apt-key add -
source /etc/lsb-release
echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
sudo apt-get update && sudo apt-get install influxdb
sudo service influxdb start

read  -n 1 -p "Setup the InfluxDB and Prometheus" mainmenuinput # TODO: automate this at some point.