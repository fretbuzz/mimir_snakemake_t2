#!/usr/bin/env bash
wget https://ftp.mozilla.org/pub/firefox/releases/63.0/linux-x86_64/en-US/firefox-63.0.tar.bz2
tar -xvf firefox-63.0.tar.bz2
cd ./firefox/
sudo cp -R . /usr/local/bin/
wget https://github.com/mozilla/geckodriver/releases/download/v0.24.0/geckodriver-v0.24.0-linux64.tar.gz
tar -xvf geckodriver-v0.24.0-linux64.tar.gz 
cp ./geckodriver /usr/local/bin/
pip pip install selenium
firefox --version #to check that it is worknig
