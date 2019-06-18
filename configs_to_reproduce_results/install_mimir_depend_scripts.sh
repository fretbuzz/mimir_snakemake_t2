#!/usr/bin/env bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Second, install Steel Bank Common Lisp (SBCL)
sudo aptitude update
sudo aptitude install sbcl -y

## TODO: there's a problem here... we need it to work via script but the sbcl
## stuff requires typing on a terminal...
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
# maybe try wihout th load?
sbcl --dynamic-space-size 2560 --load quicklisp.lisp --script ../configs_to_reproduce_results/sbcl_script2.lisp
#(quicklisp-quickstart:install)
#0
#(ql:quickload :clml :verbose t)
#(exit)

# Third, tshark and editcap are needed to process the collected PCAP file.
sudo aptitude install wireshark-common -y
sudo aptitude install tshark -y

# Fourth, pdfkit is used for display purposes.
sudo aptitude install wkhtmltopdf -y
sudo cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf

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
