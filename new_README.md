# MIMIR: Graph-based Data Exfiltration Detection for Microservices

(todo: add sub-links to the usage part of the TOC)
## Table of Contents
+ [About](#about)
+ [Getting Started](#getting_started)
+ [Usage](#usage)
+ [FAQ](#FAQ)
+ [Reproducing Graphs from Paper](#repro)

## About <a name = "about"></a>
This repository contains a prototype implementation of a graphical method for detecting data exfiltration in 
microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants. 
The goal is to detect data exfiltration, but it should also be effective at detecting other types of anomalous traffic, such as port scans or lateral movements by attackers.

This analysis pipeline been tested on Ubuntu 16.04. 

## Getting Started <a name = "getting_started"></a>
These instructions will walk you through setting up the analysis pipeline and running the demo. The demo consists of a
pre-trained model and some provided network activity data. 

See [Reproducing Graphs from Paper](#repro) for how to reproduce the 
graphs from the paper draft. See [Collecting New Data](#collecting data) for how to collect additional
network activity data and [Training New Model](#train_new_model) for how to train a new model.

### Prerequisites
This section will install the dependencies needed for running on the analysis pipeline on Ubuntu 16.04.

###### Step 0: Clone the repo and move to the analysis_pipeline directory
The following steps assume that you have cloned the repo and moved to the mimir_v2/analysis_pipeline/ directory.
```angular2html
git clone https://github.com/fretbuzz/mimir_v2.git
cd mimir_v2/analysis_pipeline
```

###### Step 1: Install Docker
[Docker](https://docs.docker.com/install/) is needed because the system uses the MulVal container.
Using the Docker convience installation script, Docker can be installed by:

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

###### Step 2: Install SBCL (Steel Bank Common Lisp)
[SBCL](http://www.sbcl.org/getting.html) is needed if you want to compare to the
[directional eigenvector method](http://ide-research.net/papers/2004_KDD_Ide_p140.pdf), though we just compare the 
angle to a fixed threshold. SBCL can be installed via
```
sudo aptitude update
sudo aptitude install sbcl -y
```

###### Step 2a: Install Quicklisp (Lisp package manager)

```angular2html
# need to move to the mimir_v2/analysis_pipeline/ directory and then run:
curl -O https://beta.quicklisp.org/quicklisp.lisp
curl -O https://beta.quicklisp.org/quicklisp.lisp.asc
sbcl --load quicklisp.lisp --script ../configs_to_reproduce_results/sbcl_script1.lisp
```

###### Step 2b: Install Common Lisp Machine Learning Library
```
git clone https://github.com/mmaul/clml.git
mv ./clml ~/quicklisp/local-projects/
sbcl --dynamic-space-size 2560 --load quicklisp.lisp --script ../configs_to_reproduce_results/sbcl_script2.lisp
```

###### Step 3: Install tshark and editcap
[Tshark \& editcap](https://www.wireshark.org/docs/wsug_html_chunked/ChapterBuildInstall.html) are used to parse 
the relevant pcap files. The first set of commands exist to avoid the interactive screen where you decide if 
non-superusers should be able to capture packets (is set to false below).
```
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install wireshark
echo "wireshark-common wireshark-common/install-setuid boolean false" | sudo debconf-set-selections
sudo DEBIAN_FRONTEND=noninteractive dpkg-reconfigure wireshark-common

sudo aptitude install wireshark-common -y
sudo aptitude install tshark -y
```

###### Step 4: Install pdfkit
[Pdfkit](https://github.com/pdfkit/pdfkit/wiki/Installing-WKHTMLTOPDF) is used to generate reports, which at the current
 stage is the best way to view the system's performance. Note: the system assumes that wkhtmltopdf is located at 
 /usr/local/bin/wkhtmltopdf. You might have to move it there on some systems, such as Ubuntu 16.04.
```
sudo aptitude install wkhtmltopdf -y

# only needed if it is not automatically located at /usr/local/bin/wkhtmltopdf
# (must do on Ubuntu 16.04)
sudo cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf
```

###### Step 5: Install Python
Make sure [Python 2.7](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/) are installed.
```
sudo aptitude install python2.7 -y
sudo apt-get install python-dev -y
wget https://bootstrap.pypa.io/get-pip.py
sudo python2 get-pip.py
```

###### Step 6: Install Python-related dependencies
Many python packages must be installed

```
# make sure pip is up to date
sudo pip install pip --user

# then install most of the python dependencies
sudo pip install docker networkx matplotlib jinja2 pdfkit numpy pandas seaborn Cython pyyaml multiprocessing scipy pdfkit tabulate sklearn --user

# on Ubuntu 16.04, pygraphviz needs to be installed in a specific way.
sudo apt-get install -y graphviz libgraphviz-dev pkg-config
sudo pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"

```

###### Step 6: Run the demo
To verify that everything is working, the demo can be run. The demo
consists of a pre-trained model plus some provided network activity data.

```angular2html
# make sure you are still in the mimir_v2/analysis_pipeline directory

# then get the example demo
git clone https://github.com/fretbuzz/mimir_example_data.git

# then unzip the provided network activity data
cd ./mimir_example_data/
gzip -d wordpress_example_final.pcap.gz
cd ..

# finally run the demo
sudo python mimir.py --training_config_json mimir_example_data/wordpress_model/example_wordpress_exp_config.json --eval_config_json mimir_example_data/wordpress_example_experiment.json
```

###### Step 7: Examine the Output
Go the mimir_example_data/results directory. There will be a LOT of generated files there, but please focus your 
attention on the PDF files. These contain reports describing system perform, including ROC curves, per-path confusion 
matrices, and descriptions of the model coefficients. The CSV files can also be useful for debugging purposes 
(or if you want to run your own statistical analysis, e.g. in R)'.

## Usage <a name = "usage"></a>
(TODO: add hyperlinks in this first paragraph)

The general system workflow for this research prototype involves first collecting some microservice network activity data, 
then training a model on part of this data, and finally applying this model to some other network activity data.

### Collecting New Data <a name = "collecting data"></a>
Collecting new data involves deploying a single-node Kubernetes cluster, deploying a microservice application on top of it,
and then collecting network activity data while simulating user traffic. The network activity data is collected by running
tcpdump on the 'any' interface of the 'default' namespace of the node, while recording which cluster entities correspond 
to which IP addresses (since there is no time to resolve hostnames while running tcpdump). In other words, the two pieces 
of necessary information to be collected are (1) the network PCAP (w/o resolving hostnames) and (2) the log mapping 
IP addresses to cluster entities.

The experimental_apparatus will handle collecting those two pieces of information. It also helps with deploying/scaling
the microservice application and simulating user traffic. Before using the experimental_apparatus component, it is necessary to install/start the Minikube single-node Kubernetes 
cluster, install the system dependencies, and setup the configuration files correctly.

The following video goes through this process on a new Ubuntu 16.04 VM (script can be found \[TODO: update script and insert it here\]):

\[TODO: update video and insert it here\]

Note that in a production setup, using a single-node Kubernetes cluster is unlikely to be desired and running tcpdump 
is a bad use of system resources. These two assumptions exist in the system currently due to time constraints. At the first
available opportunity, the system will be modified to work with a multi-node Kubernetes cluster, because this will
require modifications to the graph-based representation of network communicationn.

### Training New Model <a name = "train_new_model"></a>
\[TODO: add instructions for training new model \] 
(assume (1) that they already have acquired the data by this stage (b/c that's in the previous step), 
        (2) assume they can read the Wiki for the full instructions (so reference an example config file and explain
         how to verify))

### Run Model on New Data <a name = "run_on_new_data"></a>
\[TODO: add instructions for applying model to new data\] 
(make the same assumptions as the previous step)

## FAQ <a name = "FAQ"></a>
\[TODO: fill out the parts below (no FAQ-section-summary is needed )\]

#### How to Configure System Environment for Running the Program

#### How to setup systems to run

#### How to generate benign/attack data

#### Reproducing Graphs from Paper <a name = "repro"></a>
\[TODO -- just say how to use the scripts + give hardware specs\]

#### Why is this README being updated so slowly?

I have a significantly-over-full-time job that is completely unrelated to this.
