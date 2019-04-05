#!/usr/bin/env bash

git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/server/
sudo su
apt-get install ruby-dev
apt-get install gem
gem install bundler
bundle install