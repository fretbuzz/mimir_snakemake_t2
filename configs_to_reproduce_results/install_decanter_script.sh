#!/usr/bin/env bash
# NOTE: NEED TO BE IN # move to ./analysis_pipeline/dlp_stuff/

sudo apt-get install -y cmake make gcc g++ flex bison libpcap-dev libssl-dev python-dev swig zlib1g-dev
sudo sh -c "echo 'deb http://download.opensuse.org/repositories/network:/bro/xUbuntu_16.04/ /' > /etc/apt/sources.list.d/bro.list"
sudo apt-get update
sudo apt-get install -y bro
PATH=$PATH:/opt/bro/bin/
sudo pip install --user brothon pandas editdistance IPy networkx
git clone https://github.com/fretbuzz/decanter.git