bash ../install_docker_compose.sh
git clone https://github.com/dockersamples/atsea-sample-shop-app.git 
cd ./atsea-sample-shop-app

mkdir certs

{ echo '\n';
  echo '\n';
  echo '\n';
  echo '\n';
  echo '\n';
  echo '\n';
  echo '\n';
} | openssl req -newkey rsa:4096 -nodes -sha256 -keyout certs/domain.key -x509 -days 365 -out certs/domain.crt

sudo docker secret create revprox_cert certs/domain.crt
sudo docker secret create revprox_key certs/domain.key
sudo docker secret create postgres_password certs/domain.key
echo staging | sudo docker secret create staging_token -
sudo docker-compose up --build


