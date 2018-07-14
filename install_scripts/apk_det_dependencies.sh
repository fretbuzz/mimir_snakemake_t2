apk update
apk upgrade
apk add git
apk add python
apk add curl
apk add build-base
apk add python-dev
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
git clone https://github.com/PaulSec/DET /DET
pip install -r /DET/requirements.txt --user