sudo apt-get update
sudo apt-get install default-jdk

# TODO: setup JAVA_HOME environemntal variable
sudo echo '\nJAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java"' >> /etc/environment

source /etc/environment
echo $JAVA_HOME
sudo apt-get install -y maven
