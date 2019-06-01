## Mimir
Mimir is an experimental system to evaluate the potential for graphical, anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants. The goal is to detect data exfiltration, but it should also be effective at detecting other types of anomalous traffic, such as port scans or lateral movements by attackers.

Please see the Wiki for full documentation on how to run the experimental apparatus (to acquire sample data) and for a full explanation of how to run the analysis pipeline. The explanation below is how to run the analysis pipeline with a pretrained model, but it's likely you'd want to generate your own model.

## Running Analysis Pipeline Demo
The analysis pipeline takes a pcap file and a log of entities that exist on the cluster (such as pods and services) and uses them to generate a graph-based statistical model of network traffic on the application. This model can then be applied to new pcaps to determine if there is anomalous traffic.

This analysis pipeline been tested on MacOS 10.14.4 and Ubuntu 16.04. It will not work on Windows.

This demo will walk through installing the necessary dependencies, acquiring example data (including a pretrained model and a short pcap file), running the pipeline, and examining the output. For a full description of system components and functionality, please see the Wiki.

### Step 1: Prerequisites

[![Asciicast of setting up and running the demo](https://asciinema.org/a/249574.svg)](https://asciinema.org/a/249574)

These specific commands are for installing the system dependencies onto Ubuntu 16.04.

First, install non-python-related dependencies. 
* [Docker](https://docs.docker.com/install/) is needed because the system uses the MulVal container. Using the convience scripts, this can be installed by:
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

* [SBCL](http://www.sbcl.org/getting.html) is needed if you want to compare to [directional eigenvector method](http://ide-research.net/papers/2004_KDD_Ide_p140.pdf), though we just compare the angle to a fixed threshold. SBCL can be installed via
```
sudo aptitude install sbcl -y
```
It is necessary to install [quicklist](https://www.quicklisp.org/beta/) as the package manager, and then install the [Common Lisp Machine Learning Library](http://quickdocs.org/clml/). After SBCL is installed, the rest can be setup using this set of instructions:
```
curl -O https://beta.quicklisp.org/quicklisp.lisp
curl -O https://beta.quicklisp.org/quicklisp.lisp.asc
sbcl --load quicklisp.lisp
(quicklisp-quickstart:install)
(exit)

git clone https://github.com/mmaul/clml.git
mv ./clml ~/quicklisp/local-projects/
sbcl --dynamic-space-size 2560 --load quicklisp.lisp
(quicklisp-quickstart:install)
0
(ql:quickload :clml :verbose t)
(exit)
```
* [Tshark \& editcap](https://www.wireshark.org/docs/wsug_html_chunked/ChapterBuildInstall.html) are used to parse the pcap. 
```
sudo aptitude install wireshark-common -y
sudo aptitude install tshark -y
```

* [Pdfkit](https://github.com/pdfkit/pdfkit/wiki/Installing-WKHTMLTOPDF) is used to generate reports, which at the current stage is the best way to view the system's performance. Note: the system assumes that wkhtmltopdf is located at /usr/local/bin/wkhtmltopdf. You might have to move it there on some systems, such as Ubuntu 16.04.
```
sudo aptitude install wkhtmltopdf -y
sudo cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf
```

Then install the python-related dependencies.

* Make sure [Python 2.7](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/) are installed.
```
sudo aptitude install python2.7 -y
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py
sudo apt-get install python-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python2 get-pip.py
```


* Make sure pip is up-to-date:
```
sudo pip install pip --user
```

* Then install the necessary python packages:
```
sudo pip install docker networkx matplotlib jinja2 pdfkit numpy pandas seaborn Cython pyyaml multiprocessing scipy pdfkit tabulate sklearn --user
```
* Then install the graphviz-related dependencies:
```
sudo apt-get install -y graphviz libgraphviz-dev pkg-config
```
On Ubuntu 16.04 the following options are needed when installing pygraphviz (might not be needed on other OS's):
```
sudo pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
```

### Step 2: Clone Repo
```
git clone https://github.com/fretbuzz/mimir_v2.git
```

### Step 3: Get example data
```
cd mimir_v2/analysis_pipeline
git clone https://github.com/fretbuzz/mimir_example_data.git
cd ./mimir_example_data/
gzip -d wordpress_example_final.pcap.gz
cd ..
```

### Step 4: Starting the system
The system can be started via:
```
python mimir.py --training_config_json mimir_example_data/wordpress_model/example_wordpress_exp_config.json --eval_config_json mimir_example_data/wordpress_example_experiment.json
```

This example uses a pretrained model and detects synthetically injected attacks on a Wordpress deployment.
There is a PCAP file corresponding to this deployment in the example data. 
Note: there are no physical exfiltration events in this pcap file, i.e. all exfiltration events are simulated by the system. This is why it takes a while to run.

Of course, A new model can be also be generated based off of training data. Please see the Wiki for a full descriptin.

### Step 5: Examining the Output
Go the mimir_example_data/results directory. There will be a LOT of generated files there, but please focus your attention on the PDF files. These contain reports describing system perform, including ROC curves, per-path confusion matrices, and descriptions of the model coefficients. The CSV files can also be useful for debugging purposes (or if you want to run your own statistical analysis, e.g. in R)'.

### FAQ

#### How to Configure System Environment for Running the Program

Step 1 above walks through how to install the system dependencies on Ubuntu 16.04, but here's some additional info in list form:

* OS platforms: TODO

* Python Version: TODO

* Python2.7 Library Versions: TODO

* Docker Version: TODO

* SBCL Version: TODO

* Tshark/Editcap Version: TODO

* Pdfkit Version: TODO

#### How to setup systems to run

Follow the steps in "Step 1" above. Everything uses the latest version. These setup instructions can be copy-pasted into Ubuntu 16.04.

#### How to generate benign/attack data

It is necessary to deploy a single-node Kubernetes cluster and then deploy a microservice application ontop of it.
It is then necessary to run TCPDUMP on the 'any' interface on the default namespace of the node, while keeping track of
which entites are referred to by the IP addresses (because so many packets, that there is no time to resolve hostnames).

TODO: explain how to do all of the components...

TODO: add asciicinema video

Note that in a production setup, using a single-node Kubernetes cluster is unlikely to be desired and running tcpdump is
a bad use of system resources. These two assumptions exist in the system currently due to time constraints.

### How to verify the results

TODO: explain how to reproduce the results from the paper...

### Why is this FAQ being updated so slowly?

I have a significantly-over-full-time job that is completely unrelated to this.

### Is this project complete?

No, there are three big problems:
* It only works on single-node Kubernetes clusters. The smallest Kubernetes cluster that should be used for anything is 3 node.
* It does not properly account for workload varying over time. This causes the assumptions behind the linear model to become invalid over time, decreasing performance. It is unclear how to fix this.
* Requires running tcpdump on the 'any' interface of the 'default' namespace on each node. This is high overhead.
