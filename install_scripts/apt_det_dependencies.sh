apt-get update
export DEBIAN_FRONTEND=noninteractive
apt-get --force-yes -y install git python curl python-dev gcc
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
git clone https://github.com/PaulSec/DET /DET
pip install -r /DET/requirements.txt --user