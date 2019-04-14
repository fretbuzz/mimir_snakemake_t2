NOTE: I'll finish giving a run-down of how to run the code + some example data on 4/15.

## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants.

Be advised that the project is still pre-alpha and does not do everything that you'd want it to.


## Running Analysis Pipeline
The analysis pipeline takes pcap files and generates a trained model for detecting anomalies on further traffic from that applicaton. Currently, it ONLY generates the model (using the training set) and gives results on the validation set, at the various hyperparameter options. It WILL also be able to apply this trained model to a testing/eval set SOON, but it doesn't do that YET (should be working by ~4/15).

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
\[ TODO \]

### Step 3: Set configuration file
We'll use the analysis_pipeline/analysis_json/sockshop_example.json configuration file. If you copied the example data into the analysis_pipeline/ directory, as explained in step two, then you do NOT need to do anything and can move to step 3. Else, you'll need to modify some parts of the configuration file, as explained in the "Analysis Pipeline configuration parameters" page in the wiki.

This is most useful when running for the first time. On later times (i.e. if re-running), you can skip some parts of the analysis pipeline by setting the parameters of the configuration file appropriateley (as explained in the aforementioned wiki page).

### Step 4: Starting the system
Move to the analysis_pipeline/ directory. The system can be started via:
```
python mimir.py --training_config_json analysis_json/sockshop_example.json
```

The system will LATER be also able to take an evaluation configuration file, to give online alerts. But it CANNOT do that right now. But it WILL be able to do that SOON (by ~4/15).

### Step 5: Examining the Output
\[TODO\]
