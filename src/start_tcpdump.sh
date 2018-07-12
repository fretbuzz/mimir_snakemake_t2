#!/bin/bash

# start_tcpdump - starts one or more special docker containers (that can perform tcpdump), switches them
# into the correct network namespace, and then starts tcpdump

network_namespace = $1
tcpdump_time = $2
orchestrator = $3
file_name = $4

docker run -it --rm -v /var/run/docker/netns:/var/run/docker/netns -v /home/docker:/outside --privileged=true nicolaka/netshoot

nsenter --net=/var/run/docker/netns/${network_namespace} sh

if [ $orchestrator == "docker_swarm" ]
then
    interface = "br0"
else
    if [ $orchestrator == "kubernetes" ]
    then
        interface = "docker0"
    else
        interface = "br0" ## might want this to be something else
    fi
fi

tcpdump -G $tcpdump_time -W 1 -i $interface -w /outside/${file_name}

