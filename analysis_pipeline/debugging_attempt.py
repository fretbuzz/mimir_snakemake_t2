import pickle, sys
import pandas as pd
#from gen_attack_templates import post_process_mulval_result
import os, docker

def test_results_df():
    with open('./check_this.txt', 'r') as f:
        cont = f.read()

    pd_df = pickle.loads(cont)

    print "all done!"

def test_attack_template_generation():
    client = docker.from_env()
    client.containers.list()
    cwd = os.getcwd()
    print "cwd", cwd
    print "mulval_related_dir", cwd + '/mulval_inouts'
    # note: going to start container running everytime (and later on will detelete it...)
    container = client.containers.run("risksense/mulval", detach=True,
                                      volumes={cwd + '/mulval_inouts': {'bind': '/mnt/vol2', 'mode': 'rw'}})

if __name__=="__main__":
    print "RUNNING"
    print sys.argv

    test_attack_template_generation()