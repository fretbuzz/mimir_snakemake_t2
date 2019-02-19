#!/usr/bin/env bash
wget https://ftp.mozilla.org/pub/firefox/releases/62.0/linux-x86_64/en-US/firefox-62.0.tar.bz2
tar -xvf firefox-62.0.tar.bz2
cd ./firefox/
sudo cp -R . /usr/local/bin/
wget https://github.com/mozilla/geckodriver/releases/download/v0.20.1/geckodriver-v0.20.1-linux64.tar.gz
tar -xvf geckodriver-v0.20.1-linux64.tar.gz 
cp ./geckodriver /usr/local/bin/
pip install selenium
firefox --version #to check that it is worknig
