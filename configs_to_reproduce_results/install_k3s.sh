sudo apt update
sudo apt -y install snapd
sudo snap install hello-world
sudo /snap/bin/hello-world

sudo snap install multipass --classic

sleep 15

sudo /snap/bin/multipass launch --name k3s-master --cpus 1 --mem 2G --disk 3G | tee /dev/null
sudo /snap/bin/multipass launch --name k3s-worker1 --cpus 4 --mem 6G --disk 5G | tee /dev/null
sudo /snap/bin/multipass launch --name k3s-worker2 --cpus 4 --mem 6G --disk 5G | tee /dev/null

# Deploy k3s on the master node
sudo /snap/bin/multipass exec k3s-master -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_KUBECONFIG_MODE="644" sh - --no-deploy=traefik"
# Get the IP of the master node
K3S_NODEIP_MASTER="https://$(sudo /snap/bin/multipass info k3s-master | grep "IPv4" | awk -F' ' '{print $2}'):6443"
# Get the TOKEN from the master node
K3S_TOKEN="$(sudo /snap/bin/multipass exec k3s-master -- /bin/bash -c "sudo cat /var/lib/rancher/k3s/server/node-token")"
# Deploy k3s on the worker node
sudo /snap/bin/multipass exec k3s-worker1 -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_TOKEN=${K3S_TOKEN} K3S_URL=${K3S_NODEIP_MASTER} sh -"
# Deploy k3s on the worker node
sudo /snap/bin/multipass exec k3s-worker2 -- /bin/bash -c "curl -sfL https://get.k3s.io | K3S_TOKEN=${K3S_TOKEN} K3S_URL=${K3S_NODEIP_MASTER} sh -"

sudo /snap/bin/multipass list
sudo /snap/bin/multipass exec k3s-master kubectl get nodes

# now there should be 3 vms up and running... next step is to make it a kubernetes cluster for realsies

curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
curl -sfL https://get.k3s.io | K3S_KUBECONFIG_MODE="644" sh -s -

# Copy the k3s kubectl config file locally
mkdir ${HOME}/.kube # I added this...
sudo /snap/bin/multipass copy-files k3s-master:/etc/rancher/k3s/k3s.yaml ${HOME}/.kube/k3s.yaml
# Edit the kubectl config file with the right IP address
sed -ie s,https://127.0.0.1:6443,${K3S_NODEIP_MASTER},g ${HOME}/.kube/k3s.yaml
# Check
kubectl --kubeconfig=${HOME}/.kube/k3s.yaml get nodes
# so we don't need to specify kubeconfig flag everytime...
export KUBECONFIG=${HOME}/.kube/k3s.yaml

wget https://get.helm.sh/helm-v2.16.1-linux-amd64.tar.gz
tar -zxvf helm-v2.16.1-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm

kubectl --kubeconfig=${HOME}/.kube/k3s.yaml -n kube-system create serviceaccount tiller

kubectl --kubeconfig=${HOME}/.kube/k3s.yaml create clusterrolebinding tiller \
--clusterrole=cluster-admin \
--serviceaccount=kube-system:tiller

helm --kubeconfig=${HOME}/.kube/k3s.yaml init --service-account tiller

# if wanted to test if helm worked, then should try this:
# helm --kubeconfig=${HOME}/.kube/k3s.yaml install stable/mysql
