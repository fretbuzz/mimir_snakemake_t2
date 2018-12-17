#bash ../install_docker_compose.sh
git clone https://github.com/micronaut-projects/micronaut-examples.git 
cd ./micronaut-examples/petstore/
./gradlew build -x test # and wait a long time
sudo docker-compose build
sudo docker-compose up -d
# note: if comments not working, then close and start again!
