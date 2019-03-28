#!/usr/bin/env bash

# first install skaffold
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
chmod +x skaffold
sudo mv skaffold /usr/local/bin

# second clone relevant directory + switch into it
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
cd ./microservices-demo

# third, deploy using skaffold
skaffold run
# vv is kinda outdated and has some problems (such as the cc # in load generator being EXPIRED)
#kubectl apply -f ./release/kubernetes-manifests.yaml

# fourth, scale (either set or with auto)
## TODO