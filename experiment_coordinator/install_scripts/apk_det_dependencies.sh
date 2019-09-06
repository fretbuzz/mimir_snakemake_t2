cd /
apk update
#apk upgrade
apk add git
apk add python
apk add curl
apk add build-base
apk add python-dev
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py -k
python get-pip.py
export GIT_SSL_NO_VERIFY=1
git clone https://github.com/fretbuzz/DET /DET
git -C /DET pull
pip2 install 'certifi==2015.4.28' --force-reinstall
pip install requests
pip install -r /DET/requirements.txt --user

pip install pycrypto
if [ $? -eq 0 ]; then
    echo OK
else
    pip install pycryptodome
fi



git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/client/
make

# now here's some code to reduce the size of the resulting image
rm -rf /var/lib/apt/lists/*