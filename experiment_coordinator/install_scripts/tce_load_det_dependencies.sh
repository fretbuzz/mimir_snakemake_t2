#!/usr/bin/env bash
tce-load -wi git
tce-load -wi python
tce-load -wi compiletc
tce-load -wi python-dev
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --user
git clone https://github.com/fretbuzz/DET /DET
git -C /DET pull
pip install -r /DET/requirements.txt --user