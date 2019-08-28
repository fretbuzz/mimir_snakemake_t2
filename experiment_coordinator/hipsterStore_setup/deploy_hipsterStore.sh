#!/usr/bin/env bash

# first install skaffold
### commented out b/c it'll be done by run_exp_on_cloudlab.py
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
chmod +x skaffold
sudo mv skaffold /usr/local/bin

# second clone relevant directory + switch into it

git clone https://github.com/GoogleCloudPlatform/microservices-demo.git || true
echo "still going"
cd ./microservices-demo

# third, deploy using skaffold
## skaffold run
# they fixed the pre-built images, so those should be used instead....
sudo kubectl apply -f ./release/kubernetes-manifests.yaml

#kubectl delete deploy loadgenerator
