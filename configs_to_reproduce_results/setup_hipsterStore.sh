#!/usr/bin/env bash
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
chmod +x skaffold
sudo mv skaffold /usr/local/bin
cd /mydata/mimir_v2/experiment_coordinator
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
cd ./microservices-demo
sudo skaffold run
cd ..