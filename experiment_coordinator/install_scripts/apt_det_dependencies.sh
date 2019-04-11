cd /
apt-get -o Acquire::ForceIPv4=true update
export DEBIAN_FRONTEND=noninteractive
apt-get —no-install-recommends --force-yes -y -o Acquire::ForceIPv4=true install git python curl python-dev gcc make
curl --ipv4 https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
set GIT_SSL_NO_VERIFY=true
git clone https://github.com/fretbuzz/DET /DET
pip install -r /DET/requirements.txt --user
git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/client/
make

rm -rf /var/lib/apt/lists/*