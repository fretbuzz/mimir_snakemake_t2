## Mimir
Munnin is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. There are instructions in the Wiki on how to setup some example microservice-architecture applications. Then run_experiment can be used to coordinate the generation of background traffic with exfiltration of data. pcap_parser can can be used to analyze the resulting pcaps. It parses the pcaps into a graph-representation, saves the edgefiles, calculates some graph metrics, saves those too, and then makes some graphs. 

## Motivation
The transition to microsevice-architectures bring new challenges and opportunities to detecting data exfiltration. Current data exfiltration methods use keyword-based methods that don't work on encrypted traffic, necessitating the use of anomaly-based methods. Anomaly-based methods, however, don't work well in tradtional enterprise applications because of the lack of network-level visibility. Network-level visibility is increased in microservice-architecture applications, so it is worth re-checking the effectivness of network-anomaly-based methods in this new context

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

## Code Example
Just set up an experimental configuration file and set the flags correctly in run_experiment.py and you are ready to go!

## Installation
Refer to the instructions in the Wiki about how to deploy/install an application along with Mimir.

## Tests
Are found in the test.py file.

## Credits
[todo]
