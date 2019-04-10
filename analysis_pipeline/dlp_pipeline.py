###
'''
The purpose of this file is to run data through the data exfiltration detection pipeline.
Namely, this'll be DECANTeR at first, but hopefully also DUMONT later one.

At first, make this a STANDALONG script. Later one, we'll make it an integral part of the system.
'''

''' ## This is what needs to happen first.
DLP Pipeline (DECANTeR)
(0) clone DECANTeR directory into DLP_stuff (if it does not already exist)
(1) Generate Bro log files
    this'll involve calling bro on the relevant pcap and then storing the resulting logs in a nice locaiton
(2) "Live" Analysis:
	python2 main.py --training test-data/malware/vm7_decanter.log --testing test-data/malware/exiltration_logs/URSNIF_386.pcap.decanter.log -o 0
For DUMONT, we'd need to write a function that interacts with their
	python functions (so can't handle it all via the terminal)

Okay, well DECANTER looks actually pretty easy... Steps (1):
(1) Create new directory for this stuff
	clone appropriate github repo
(2) Setup Docker container w/ the new directory as a shared folder
	this'll be just like MULVAL
	    ## maybe this one??
	        https://github.com/blacktop/docker-bro
(3) Generate bro logs
(4) Generate DECANTER output
'''