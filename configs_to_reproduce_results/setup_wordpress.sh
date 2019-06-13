## WORK IN PROGRESS ##
#!/usr/bin/env bash
bash install_scripts/install_selenium_dependencies.sh
docker pull nicolaka/netshoot
wget https://get.helm.sh/helm-v2.14.1-linux-amd64.tar.gz
tar -zxvf helm-v2.14.1-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm
sudo python wordpress_setup/scale_wordpress.py 7
MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used

host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"
python wordpress_setup/setup_wordpress.py $MINIKUBE_IP $WORDPRESS_PORT "hi"

## TODO: there is more todo here! we nned to make sure that selenium works and that the rest goes through...
