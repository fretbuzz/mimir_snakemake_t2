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

##### Setting up and Running the System to Collect data

The following video goes through this process on a new Ubuntu 16.04 VM (script can be found \[TODO: update script and insert it here\]):

\[TODO: update video and insert it here\]

Note that in a production setup, using a single-node Kubernetes cluster is unlikely to be desired and running tcpdump 
is a bad use of system resources. These two assumptions exist in the system currently due to time constraints. At the first
available opportunity, the system will be modified to work with a multi-node Kubernetes cluster, because this will
require modifications to the graph-based representation of network communicationn.

##### Modifying Experiment Parameters
Please see the [corresponding wiki page](https://github.com/fretbuzz/mimir_v2/wiki/Performing-Experiment) for a description
on how to change the various parameters (e.g., application, time, load, etc.). Please see the 
[data collection config files for the data used in the paper](https://github.com/fretbuzz/mimir_v2/tree/master/configs_to_reproduce_results/Data_Collection)
for examples of how to set the config files. Please note that some applications require additional
steps before starting data collection, which is explained in the aforementioned wiki page.

### Training New Model <a name = "train_new_model"></a>
Once network activity data has been collected, a new model can be trained. To do this, a corresponding configuration file 
must be created. This can be done by modifying the corresponding 
[example config file](https://github.com/fretbuzz/mimir_v2/blob/master/analysis_pipeline/analysis_json/training_config_example.json).
Change the "exp_config_file" field to point to the *_analysis.json file created when the corresponding data was collected.
This will be in the directory that was created during data collection. For a full explanation of all the fields, please
see the corresponding [wiki page](https://github.com/fretbuzz/mimir_v2/wiki/Run-Analysis-Pipeline).

Then the new model can be generated by running
```angularjs
python mimir.py --training_config_json [path_to_training_config_file]
```

After training the model, you probably want to go back to the corresponding configuration file and change the "get_endresult_from_memory"
field to "true". This allows the model to be applied to new data without being re-generated in the process.

### Run Model on New Data <a name = "run_on_new_data"></a>
The model can be applied to new data in two days; either exfiltration can be simulated by the system or it can attempt 
to detect live exfiltration.

#### Simulating Exfiltration
When running the system in this mode, exfiltration is simulated in the same way that it is when
training a model. This mode is useful for e.g., testing new algorithms because
it can evaluate several different exfiltration rates easily.

A corresponding configuration file must be created. This can be done by modifying the corresponding 
[example config file](mimir_v2/analysis_pipeline/analysis_json/testing_config_example.json).
Change the "exp_config_file" field to point to the *_analysis.json file created when the corresponding data was collected.
This will be in the directory that was created during data collection. For a full explanation of all the fields, please
see the corresponding [wiki page](https://github.com/fretbuzz/mimir_v2/wiki/Run-Analysis-Pipeline). 

Please note that you probably should (but do not need to) change the "cur_experiment_name" field too. 
This is because, when analyzing the collected data, each analysis is cached using the "cur_experiment_name" in the config file.
If you attempt to perform two analyses using the same "cur_experiment_name", then the former analysis will be overwritten by the latter.

```angularjs
python mimir.py --training_config_json [path_to_training_config_file] --eval_config_json [path_to_testing_config_file]
```

#### Detecting Live Exfiltration
This mode looks for live exfiltration; it does not simulate exfiltration. It returns a list of scores indicating how likely
data exfiltration occured during the corresponding time interval. These scores are between 0 and 1, and the cutoff point
"should" be 0.5, though in the current system this is not always optimal (due to the problems with modeling changing
workloads). 

Setting up the config file is the same as it is in the "Simulating Exfiltration" mode. However, the command to run the system is slightly
different:
```angularjs
python mimir.py --live --training_config_json [path_to_training_config_file] --eval_config_json [path_to_testing_config_file]
```

Please note that the system will ignore all the fields in the config file related to data exfiltration simulation; please
see the corresponding [wiki page](https://github.com/fretbuzz/mimir_v2/wiki/Run-Analysis-Pipeline) for more information.

## FAQ <a name = "FAQ"></a>
\[TODO: fill out the parts below (no FAQ-section-summary is needed )\]

#### How to Configure System Environment for Running the Program
[Getting Started](#getting_started) walks through how to install the system dependencies on Ubuntu 16.04, but here's 
some additional information on the specific software versions required:

* OS platforms: Ubuntu 16.04 (also regularly used on MacOS 10.14.4)

* Python Version: 2.7.12

* Python2.7 Dependency Versions: See analysis_pipeline/requirements.txt [todo: update this! using the new install script (literally just run it and then pip freeze)]

* Docker Version: 18.09.6

* SBCL Version: 1.3.1.debian

* Tshark/Editcap Version: 2.6.8

* wkhtmltopdf Version: 0.12.2.4

#### How to setup systems to run
Please refer to [Getting Started](#getting_started).

#### How to generate benign/attack data
**How to generate benign data:**
Please see [Collecting New Data](#collecting data) for an explanation of how to collect data. The system does not *currently*
support performing live exfiltration on the deployed cluster, so all collected data will be benign data.

**How to generate attack data:** Synthetic attack data is generated by the analysis_pipeline during normal operation. 
So just running the system as normal will generate the synthetic attack data as a byproduct.
Running physical exfiltration events on the deployed application is currently *not* (fully) supported by the 
experimental_apparatus component, though it should be eventually.

#### Reproducing Graphs from Paper <a name = "repro"></a>
[TODO: explain how to reproduce the results from the paper  just say how to use the scripts + where to find graphs + give hardware specs\]

Reproducing the paper's results consists of two parts:

* First, data similar to that used in the paper
must be acquired. However, the three main results require ~100GB of data each. This is too difficult to
distribute, so this data must be re-collected from a local Kubernetes cluster (see [Collecting New Data](#collecting data)
for more information). The config files used to generate the data used in the paper are provided in the 
mimir_v2/configs_to_reproduce_results/Data_Collection directory. A convenience script to collect this data
is provided and described below.

* Second, this data must be processed in the same way that the data in the paper is processed. [TODO: keep explaining this part]
[todo: please be advised that each of these thigns takes 3-4 days on a server w/ the hardware
 specs described below]

```
## [TODO: Finish/explain this part]
# maybe git clone here
cd ./mimir_v2/experiment_coordinator/ # need to be in the experiment_coordinator directory
# second, run ONE of these:
. ../configs_to_reproduce_results/e2e_repro_scripts/sockshop_repro.sh
. ../configs_to_reproduce_results/e2e_repro_scripts/sockshop_repro_PROB_DISTRO.sh
. ../configs_to_reproduce_results/e2e_repro_scripts/wordpress_repro.sh
```


Hardware used in eval setup: [TODO]

#### Why is this README being updated so slowly?

I have a significantly-over-full-time job that is completely unrelated to this.