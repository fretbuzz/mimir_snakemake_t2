# this function parses the terminal output to determine the the actual approximate rate that data
# exfiltration takes during an experiment

# usage: python parse_terminal_output.py [file of terminal output to parse]

import sys

if __name__=="__main__":
    print "RUNNING"

    file_path = sys.argv[1]
    pod_ip_file = open(file_path, "r") # result of kubernetes get po -o wide (--all-namespaces)
    if pod_ip_file.mode == 'r':
        contents = pod_ip_file.read()
        print contents

    # okay, time to parse this...

