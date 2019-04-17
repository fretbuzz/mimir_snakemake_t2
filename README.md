## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants. The goal is to detect data exfiltration, but it should also be effective at detecting other types of anomalous traffic, such as port scans or lateral movements by attackers.

Be advised that the project is still pre-alpha, does not do everything that you'd want it to, and the interface is a little rough. I recommend checking back ~4/20, which is when I hope to have it ironed out. 

## Running Analysis Pipeline
The analysis pipeline takes a pcap file and a log of entities that exist on the cluster (such as pods and services) and uses them to generate a graph-based statistical model of network traffic on the application. This model can then be applied to new pcaps to determine if there is anomalous traffic.

This analysis pipeline been tested on MacOS and Linux 16.04. It will not work on Windows.

### Step 1: Prerequisites
First, install non-python-related dependencies. 
* [Docker](https://docs.docker.com/install/) is needed because the system uses the MulVal container. 
* [SBCL](http://www.sbcl.org/getting.html) is needed if you want to compare to [directional eigenvector method](http://ide-research.net/papers/2004_KDD_Ide_p140.pdf). 
* [Tshark \& editcap](https://www.wireshark.org/docs/wsug_html_chunked/ChapterBuildInstall.html) are used to parse the pcap. 
* [Pdfkit](https://github.com/pdfkit/pdfkit/wiki/Installing-WKHTMLTOPDF) is used to generate reports, which at the current stage is the best way to evaluate performance.

Then install the python-related dependencies.

* Make sure [Python 2.7](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/) are installed.

* Make sure pip is up-to-date:
```
pip install pip
```

* Then install the necessary python packages:
```
pip install docker networkx matplotlib jinja2 pdfkit numpy pandas seaborn Cython pyyaml multiprocessing scipy pdfkit tabulate
```

### Step 2: Get example data
```
cd mimir_v2/analysis_pipeline
git clone https://github.com/fretbuzz/mimir_example_data.git
cd ./mimir_example_data/
gzip -d example_wordpress.pcap.gz
cd ..
```

### Step 3: Starting the system
The system can be started via:
```
python mimir.py --training_config_json analysis_json/wordpress_model.json --eval_config_json analysis_json/wordpress_example.json
```

This uses a pretrained model and detects synthetically injected attacks on a Wordpress deployment. There is a pcap corresponding to this deployment in the example data. Note: the system does NOT currently give alerts based on physical abnormalities in the pcap, it only runs in "synthetic injection mode", for research purposes. It will SOON give alerts based on physical abnormalities (~4/20).

A new model can be generated based off of training data (see corresponding wiki page). Be advised that it requires a pcap of network activity and a time-alignd log of entities on the kubernetes cluster (e.g. pods & services). It is likely the log of cluster entites will need to be generated from other data sources (e.g., prometheus database, etc); see the corresponding wiki page for more information.

### Step 4: Examining the Output
\[TODO\]
