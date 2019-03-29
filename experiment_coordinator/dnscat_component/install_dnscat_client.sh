#!/usr/bin/env bash

apt-get -o Acquire::ForceIPv4=true update
apt-get -o Acquire::ForceIPv4=true install git gcc make
git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/client/
make