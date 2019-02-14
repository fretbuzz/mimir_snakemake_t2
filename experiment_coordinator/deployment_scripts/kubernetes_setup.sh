cd /mydata/
sudo chown -R jsev ./
set +e
if cd /mydata/mimir;
then cd ..;
else git clone https://github.com/fretbuzz/mimir_snakemake_t2;
fi

echo "more stuff"
vboxmanage setproperty machinefolder /mydata/
echo "test"
sudo usermod -aG docker $USER
echo "testtest"
newgrp docker <<EONG

# Setup KVM2 driver
echo "even more stuff"
sudo apt-get update
sudo apt install -y libvirt-bin libvirt-daemon-system qemu-kvm
sudo usermod -a -G libvirtd $(whoami)
newgrp libvirtd

# there is a new docker overlay driver, i don't like it. I'm going to switch it for a different one
echo {\"storage-driver\": \"vfs\"} > /etc/docker/daemon.json
sudo systemctl stop docker
sudo systemctl start docker

# Install KVM2 driver
echo "even even more stuff"
curl -Lo docker-machine-driver-kvm2 https://storage.googleapis.com/minikube/releases/latest/docker-machine-driver-kvm2 \
&& chmod +x docker-machine-driver-kvm2 \
&& sudo cp docker-machine-driver-kvm2 /usr/local/bin/ \
&& rm docker-machine-driver-kvm2

# start minikube
export MINIKUBE_HOME=/mydata

minikube start --vm-driver kvm2 --cpus=16 --memory=100000 --disk-size 65g

echo "At end"
EONG

