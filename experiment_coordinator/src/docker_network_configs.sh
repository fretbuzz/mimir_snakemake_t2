#!/usr/bin/env bash

# prints out some simple info about the currently running docker containers
# arg1 ($1) = name of output file to contain network_ids
# arg2 ($2) = name of output file to contain full container configs

echo "${1}"
echo "${2}"

docker network ls -q > "${1}"

while read p; do
    docker network inspect $p -v >> "${2}" ;
done < "${1}"
