git clone https://gitlab.com/charts/gitlab.git

cd ./gitlab

helm init
helm repo add gitlab https://charts.gitlab.io

cd ./gitlab
helm dependencies update

helm upgrade --install gitlab . \
  --timeout 600 \
  --set global.hosts.domain=example.local \
  --set global.hosts.externalIP=10.10.10.10 \
  --set gitlab.migrations.initialRootPassword="example-password" \
  --set certmanager-issuer.email=me@example.local
