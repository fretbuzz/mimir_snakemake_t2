bash ./install_maven.sh
sudo bash ../install_docker_compose.sh
git clone https://github.com/ewolff/microservice.git 
cd ./microservice/
cd ./microservice-demo/
mvn package
cd ..
cd ./docker
sudo docker-compose up


