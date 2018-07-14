curl -sSL https://get.docker.com/ | sh
apt-get install -yq curl jq python-pip unzip build-essential python-dev
bash install_docker_compose.sh
base=https://github.com/docker/machine/releases/download/v0.14.0 &&
  curl -L $base/docker-machine-$(uname -s)-$(uname -m) >/tmp/docker-machine &&
  sudo install /tmp/docker-machine /usr/local/bin/docker-machine
sudo bash install_minikube_dependencies.sh
pip install docker # going to use this to interact w/ the docker daemon from my
# driver program
