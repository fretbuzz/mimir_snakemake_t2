## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants.


## Motivation
TODO

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
