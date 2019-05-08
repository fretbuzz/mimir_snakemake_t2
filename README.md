NOTE: Literally updating these instructions ATM. 

## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants. The goal is to detect data exfiltration, but it should also be effective at detecting other types of anomalous traffic, such as port scans or lateral movements by attackers.

Please see the Wiki for full documentation on how to run the experimental apparatus (to acquire sample data) and for a full explanation of how to run the analysis pipeline. The explanation below is how to run the analysis pipeline with a pretrained model, but it's likely you'd want to generate your own model.

## Running Analysis Pipeline Demo
The analysis pipeline takes a pcap file and a log of entities that exist on the cluster (such as pods and services) and uses them to generate a graph-based statistical model of network traffic on the application. This model can then be applied to new pcaps to determine if there is anomalous traffic.

This analysis pipeline been tested on MacOS and Linux 16.04. It will not work on Windows.

This demo will walk through installing the necessary dependencies, acquiring example data (including a pretrained model and a short pcap file), running pipeline, and examining the output.

### Step 1: Prerequisites
First, install non-python-related dependencies. 
* [Docker](https://docs.docker.com/install/) is needed because the system uses the MulVal container. 
* [SBCL](http://www.sbcl.org/getting.html) is needed if you want to compare to [directional eigenvector method](http://ide-research.net/papers/2004_KDD_Ide_p140.pdf). 
* [Tshark \& editcap](https://www.wireshark.org/docs/wsug_html_chunked/ChapterBuildInstall.html) are used to parse the pcap. 
* [Pdfkit](https://github.com/pdfkit/pdfkit/wiki/Installing-WKHTMLTOPDF) is used to generate reports, which at the current stage is the best way to view the system's performance. Note: the system assumes that wkhtmltopdf is located at /usr/local/bin/wkhtmltopdf. You might have to move it there, e.g. with
```
cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf
```

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
* Then install the graphviz-related dependencies:
```
apt-get install -y graphviz libgraphviz-dev pkg-config
```
On linux 16.04 the following options are needed when installing pygraphviz (might not be needed on other OS's):
```
pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
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
gzip -d example_wordpress.pcap.gz
cd ..
```

### Step 4: Starting the system
The system can be started via:
```
python mimir.py --training_config_json analysis_json/wordpress_model.json --eval_config_json analysis_json/wordpress_example.json
```

This uses a pretrained model and detects synthetically injected attacks on a Wordpress deployment. There is a pcap corresponding to this deployment in the example data. Note: there are no physical exfiltration events in this pcap file-- all exfiltration events are simulated by the system. This is why it takes a while to run.

A new model can be generated based off of training data (see corresponding wiki page). Be advised that it requires a pcap of network activity and a time-alignd log of entities on the kubernetes cluster (e.g. pods & services). It is likely the log of cluster entites will need to be generated from other data sources (e.g., prometheus database, etc); see the corresponding wiki page for more information.

### Step 5: Examining the Output
Go the mimir_example_data/results directory. There should be several csv files with the metrics. There should also be several pdfs with ROC curves, descriptions of the model coefficients, and per-path detection resutls.
