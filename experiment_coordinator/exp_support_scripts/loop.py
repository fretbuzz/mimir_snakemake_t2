# the purpose of this function is to keep the experimental apparatus exfiltrating over the
# entire time period that it is supposed to. Therefore, if the function completes, we are
# going to restart it. (Note that it will ultimately be stopped when we call all the python
# instances later on)

# it takes the entire command line invocation of the function as the command line argument
# these are passed directly to suborpocess


import sys
import subprocess

if __name__=="__main__":
    print "RUNNING"
    print sys.argv[1:]

    arg_list = sys.argv[1:]
    print arg_list
    args_list = ['python', '/DET/det.py', '-c', '/config.json', '-p', sys.argv[1], '-f', sys.argv[2]]
    #args_list.append(sys.argv[1])
    #print "args_list", args_list

    while True:
        out = subprocess.call(args_list, cwd='/DET/')
        print out