#!/usr/bin/env bash

# prints out some simple info about the currently running docker containers
# arg1 ($1) = name of output file for service info

echo "${1}"

kubectl get svc --all-namespaces > "${1}"
