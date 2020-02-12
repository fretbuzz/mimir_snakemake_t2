# now setup istio...
# this is all from https://istio.io/docs/setup/install/helm/
wget https://github.com/istio/istio/releases/download/1.4.3/istio-1.4.3-linux.tar.gz
tar -zxvf istio-1.4.3-linux.tar.gz
cd istio-1.4.3
export PATH=$PWD/bin:$PATH
kubectl create namespace istio-system
helm template install/kubernetes/helm/istio-init --name istio-init --namespace istio-system | kubectl apply -f -
kubectl -n istio-system wait --for=condition=complete job --all
helm template install/kubernetes/helm/istio --name istio --namespace istio-system | kubectl apply -f -

# then verify install
kubectl get svc -n istio-system
kubectl get pods -n istio-system
