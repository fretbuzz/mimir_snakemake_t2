apk-update
apk update
apk upgrade
apk add git
apk add python
apk add curl
apk add build-base # doesn't work (tho you might suspect that it would...):
apk add gcc
apk add python-dev # need certain python header files for compiling extensions required by DET (below)
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py # this takes a while...
git clone https://github.com/PaulSec/DET /
pip install -r /DET/requirements.txt --user