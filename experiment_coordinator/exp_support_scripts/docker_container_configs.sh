#!/usr/bin/env bash

# prints out some simple info about the currently running docker containers
# arg1 ($1) = name of output file to contain container IDs
# arg2 ($2) = name of output file to contain full container configs

echo "${1}"
echo "${2}"

docker ps -q > "${1}"

while read p; do
    docker inspect $p >> "${2}" ;
done < "${1}"
