apt-get -o Acquire::ForceIPv4=true update
export DEBIAN_FRONTEND=noninteractive
apt-get --force-yes -y -o Acquire::ForceIPv4=true install git python curl python-dev gcc make
curl --ipv4 https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
git clone https://github.com/fretbuzz/DET /DET
pip install -r /DET/requirements.txt --user
