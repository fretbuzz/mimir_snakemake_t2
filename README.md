## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants.


## To Run
Currently, it only works off-line (with network pcap files). Add the desired configuration to analysis_pipeline/pipeline_recipes.py and that'll run it (see current contents of file for usage example). Note: If your thinking of running this yourself, you should probably talk to fretbuzz first.

Detailed setup/running instructions are being worked on at the moment and should be available soon. If you need to run it before then, send fretbuzz a message.

 ## Tech/framework used

<b>Built with:</b>

* Kubernetes/Docker-Swarm orchestrators

* Tcpdump for pcap generation

* Weave's Sock Shop / Docker's Atsea Shop / Wordpress Helm Chart as sample applications

* Locust for background traffic simulation

* the Data Exfiltration Toolkit (DET) for data exfiltration simulation

* Data visualization using matplotlib

* Networkx for network-analysis

* the Netshoot container for tcpdumping in the correcting network namespace

* Pexpect for handling the ssh-sing

* a whole bunch of Linux coreutils (most notably sed)

* Anaconda for other data-analysis purposes

## Tests
Are found in the test.py file.

## Credits
[todo]
