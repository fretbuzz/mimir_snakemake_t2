#!/usr/bin/env bash
#curl -fsSL https://get.docker.com -o get-docker.sh
#sudo sh get-docker.sh
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable"
sudo apt-get update
sudo apt-get -y install docker-ce=17.06.0~ce-0~ubuntu
echo "docker cmd should work..."

# Second, install Steel Bank Common Lisp (SBCL)
sudo aptitude update
sudo aptitude install sbcl -y

## TODO: i think there might still be a problem with setting up the SBCL code...
# Then the Quicklisp LISP package manager is needed
curl -O https://beta.quicklisp.org/quicklisp.lisp
curl -O https://beta.quicklisp.org/quicklisp.lisp.asc
# maybe try without the load ??
sbcl --load quicklisp.lisp --script ../configs_to_reproduce_results/sbcl_script1.lisp
#(quicklisp-quickstart:install)
#(exit)

# Quicklisp will be used to install the Common Lisp Machine Learning (CLML) Library
git clone https://github.com/mmaul/clml.git
mv ./clml ~/quicklisp/local-projects/
wget https://common-lisp.net/project/asdf/archives/asdf.lisp -O asdf.lisp
# maybe try wihout the load?
sbcl --dynamic-space-size 2560 --load quicklisp.lisp --script ../configs_to_reproduce_results/sbcl_script2.lisp
#(quicklisp-quickstart:install)
#0
#(ql:quickload :clml :verbose t)
#(exit)

# Third, tshark and editcap are needed to process the collected PCAP file.
# using the code from here to handle the wireshark-common install case: https://unix.stackexchange.com/questions/367866/how-to-choose-a-response-for-interactive-prompt-during-installation-from-a-shell
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install wireshark
echo "wireshark-common wireshark-common/install-setuid boolean false" | sudo debconf-set-selections
sudo DEBIAN_FRONTEND=noninteractive dpkg-reconfigure wireshark-common

sudo aptitude install wireshark-common -y
sudo aptitude install tshark -y

# Fourth, pdfkit is used for display purposes.
# NOTE: need to be kinda fancy so that wkhtmtltopdf with patched QT is installed
#sudo aptitude install wkhtmltopdf -y
# sudo cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf
wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz
tar xvf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz
sudo mv wkhtmltox/bin/wkhtmlto* /usr/bin/
sudo ln -nfs /usr/bin/wkhtmltopdf /usr/local/bin/wkhtmltopdf
sudo apt-get install libpng-dev

# Fifth, Python is the language that the system is written in.
sudo aptitude install python2.7 -y
sudo apt-get install python-dev -y
wget https://bootstrap.pypa.io/get-pip.py
sudo python2 get-pip.py

# Sixth, there are several Python-related dependencies, which'll be handled via PIP.
sudo pip install pip --user
sudo pip install docker networkx matplotlib jinja2 pdfkit numpy pandas seaborn Cython pyyaml multiprocessing scipy pdfkit tabulate sklearn --user

# Seventh, on Ubuntu 16.04, pygraphviz needs to be installed in a specific way.
sudo apt-get install -y graphviz libgraphviz-dev pkg-config
sudo pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
