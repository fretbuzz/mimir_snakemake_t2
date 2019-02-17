#!/usr/bin/env bash
wget https://ftp.mozilla.org/pub/firefox/releases/63.0/linux-x86_64/en-US/firefox-63.0.tar.bz2
tar -xvf firefox-63.0.tar.bz2
cd ./firefox/
sudo cp -R . /usr/local/bin/
firefox --version #to check that it is worknig