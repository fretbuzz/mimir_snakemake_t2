apt update
export DEBIAN_FRONTEND=noninteractive
apt --force-yes -y upgrade
apt --force-yes -y install git python curl python-dev
apt --force-yes -y install gcc
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py # this takes a while...
git clone https://github.com/PaulSec/DET
pip install -r ./DET/requirements.txt --user