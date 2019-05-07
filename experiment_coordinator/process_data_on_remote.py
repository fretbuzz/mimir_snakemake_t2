'''
The purpose of this file is to let data be processed on a remote host (using only my local host
to process data is just too slow)
'''

import pwnlib.tubes.ssh
from pwn import *

def install_dependencies():
    pass

def sendline_and_wait_responses(sh, cmd_str, timeout=5):
    sh.sendline(cmd_str)
    line_rec = 'start'
    while line_rec != '':
        line_rec = sh.recvline(timeout=timeout)
        if 'Please enter your response' in line_rec:
            sh.sendline('n')
        print("recieved line", line_rec)

def modify_file_path():
    pass # how to do this??? can I see sed???

## TODO: mimir_num to re-use directoryies..
def process_on_remote(remote_server_ip, remote_server_key, user, eval_dir_with_data, eval_analysis_config_file,
                      model_dir, model_analysis_config_file, skip_install=True, skip_upload = False,
                      dont_retreive_eval = False, dont_retreive_train = False, mimir_num = None):
    print "starting to run on remote..."

    # step (0): connect
    s = None
    while s == None:
        try:
            s = pwnlib.tubes.ssh.ssh(host=remote_server_ip,
                keyfile=remote_server_key,
                user=user)
        except:
            time.sleep(60)
    sh = s.run('sh')
    print "sh_created..."

    #################
    # step (2) install dependencies (see github page)
    sh.sendline("sudo sh")
    line_rec = sh.recvline(timeout=5)
    print "line_rec", line_rec

    if not skip_install:
        '''
        sudo apt-get install graphviz libgraphviz-dev pkg-config
        pip install pygraphviz --user
        pip install pygraphviz --install-option=\"--include-path=/usr/include/graphviz\" --install-option="--library-path=/usr/lib/graphviz/" 
        cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf
        '''

        #sh.sendline("aptitude update")
        print "aptitude update rec:"
        sendline_and_wait_responses(sh, "aptitude update", timeout=15)

        #sh.sendline("aptitude install wireshark-common -y")
        #line_rec = sh.recvline(timeout=25)
        print "wireshark-common rec:"
        sendline_and_wait_responses(sh, "aptitude install wireshark-common -y", timeout=30)

        print "aptitude install tshark rec:"
        sendline_and_wait_responses(sh, "aptitude install tshark -y", timeout=30)

        #sh.sendline("aptitude install wkhtmltopdf -y")
        #line_rec = sh.recvline(timeout=15)
        print "wkhtmltopdf rec:"
        sendline_and_wait_responses(sh, "aptitude install wkhtmltopdf -y", timeout=30)
        sendline_and_wait_responses(sh, "cp  /usr/bin/wkhtmltopdf  /usr/local/bin/wkhtmltopdf", timeout=30)

        #sh.sendline("aptitude install sbcl -y")
        #line_rec = sh.recvline(timeout=15)
        print "aptitude install sbcl rec:"
        sendline_and_wait_responses(sh, "aptitude install sbcl -y", timeout=30)

        print "installing pygraphviz"
        sendline_and_wait_responses(sh, "apt-get install -y graphviz libgraphviz-dev pkg-config", timeout=30)
        sendline_and_wait_responses(sh, "pip install pygraphviz --install-option=\"--include-path=/usr/include/graphviz\" "
                                        "--install-option=\"--library-path=/usr/lib/graphviz/\"", timeout=30)
    ####################

    # step (3): create relevant directory and put mimir there
    # we are going to install at ~/mimir-1/ (where the 1 should be incremented for each existing...)
    print "sending this now... ls -l"
    #sh.sendline("cd /users/jsev")
    sh.sendline("cd /mydata/")
    line_rec = sh.recvline(timeout=5)
    print "line_rec", line_rec

    #'''
    line_rec = sh.recvline(timeout=5)
    print "line_rec", line_rec

    ## TODO: parse line_rec here... (i'll just put some super simple code here now for workflow purposes)
    if mimir_num is not None:
        current_mimir_dir = mimir_num
    else:
        sh.sendline("ls -l")
        lines_rec = []
        line_rec = 'start'
        while line_rec != '':
            line_rec = sh.recvline(timeout=5)
            lines_rec.append(line_rec)

        num_lines_rec = len(lines_rec)
        existing_mimir_dirs = num_lines_rec
        current_mimir_dir = existing_mimir_dirs + 1

    '''
    sh.sendline("ls -l")
    line_rec = sh.recvline(timeout=5)
    print "line_rec", line_rec

    ## TODO: parse line_rec here... (i'll just put some super simple code here now for workflow purposes)
    num_lines_rec = len(line_rec.split("\n"))
    print "num_lines_rec",num_lines_rec
    if num_lines_rec != 2:
        exit(244) # (going to exit b/c this part isn't implemented yet...)
        pass # todo: do the stuff here (if the # of lines is 2, then there's nothing there...)
    else:
        existing_mimir_dirs = 0 # (so this'd be thing
    '''

    cur_mimir_dir_name = 'mimir-' + str(current_mimir_dir)
    print "cur_mimir_dir_name",cur_mimir_dir_name

    # step (4): put relevant data there
    # okay, so what actually is this???
        # it is the moving of two things:
            # (1) the dir with all the data
            # (2) the config file (might overlap with (1) but maybe not...)
    print "eval_dir_with_data",eval_dir_with_data
    eval_dir_with_data_name = '/' + eval_dir_with_data.split("/")[-2]
    eval_exp_config_file = cur_mimir_dir_name + eval_dir_with_data_name
    print "eval_analysis_config_file.split", eval_analysis_config_file.split("/")
    eval_config_file_name = eval_analysis_config_file.split("/")[-1]
    print "eval_config_file_name",eval_config_file_name
    model_dir_with_data_name = '/' + model_dir.split("/")[-2]
    model_exp_config_file = cur_mimir_dir_name + model_dir_with_data_name
    model_config_file_name = model_analysis_config_file.split("/")[-1]

    sh.sendline("pwd")
    print sh.recvline(timeout=5)

    if not skip_upload:
        line_rec = sh.recvline(timeout=5)
        print "line_rec", line_rec
        print "cmd_to_make_cur_mimir_dir",cur_mimir_dir_name
        sh.sendline("mkdir " +  cur_mimir_dir_name)
        sh.sendline("mkdir " +  cur_mimir_dir_name + eval_dir_with_data_name)
        print "mkdir rec_line",sh.recvline(timeout=5)
        # (1) the EVAL dir with all the data
        print "eval_exp_config_file", eval_exp_config_file
        #### TODO TODO TODO TODO TODO TODO TODO TODO ####
        #sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 ~", timeout=5)
        #sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 " + cur_mimir_dir_name, timeout=5)
        #sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 " + cur_mimir_dir_name + eval_dir_with_data_name, timeout=5)
        sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 /mydata/", timeout=5)
        sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 /mydata/" + cur_mimir_dir_name, timeout=5)
        sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 /mydata/" + cur_mimir_dir_name + eval_dir_with_data_name, timeout=5)
        #s.upload(eval_dir_with_data, remote="~/" + eval_exp_config_file)
        s.upload(eval_dir_with_data, remote="/mydata/" + eval_exp_config_file) #eval_exp_config_file)
        # (2) the config file (might overlap with (1) but maybe not...)
        try:
            s.upload(eval_analysis_config_file, remote= "/mydata/" + cur_mimir_dir_name)
        except:
            print "eval_analysis_config_file probably already exists..."

        # (3) now repeat with the MODEL dir
        if eval_dir_with_data != model_dir or eval_analysis_config_file != model_analysis_config_file:
            print "try_making_this_dir:", cur_mimir_dir_name + model_dir_with_data_name
            sh.sendline("mkdir " +  cur_mimir_dir_name + model_dir_with_data_name)
            sendline_and_wait_responses(sh, "sudo chown -R jsev:dna-PG0 " + cur_mimir_dir_name + model_dir_with_data_name, timeout=5)
            s.upload(model_dir, remote= "/mydata/" + model_exp_config_file)
            try:
                s.upload(model_analysis_config_file, remote= "/mydata/" + cur_mimir_dir_name)
            except:
                print "eval_analysis_config_file probably already exists..."

    ## TODO: Okay, once I get to here, test everything above...

    # step (5): install python dependencies...
    if not skip_install:
        print "pip install pip rec:"
        sendline_and_wait_responses(sh, "pip install pip --user", timeout=30)
        print "pip install python depend encies rec:"
        sendline_and_wait_responses(sh, "pip install docker networkx matplotlib jinja2 pdfkit numpy pandas seaborn Cython \
                                    pyyaml multiprocessing scipy pdfkit tabulate --user", timeout=30)

    ##  ACTUALLY START IT
    # step (6): now actually start the system...
    #sh.sendline(cur_mimir_dir_name )
    sh.sendline("cd " + cur_mimir_dir_name)
    sendline_and_wait_responses(sh, "ls", timeout=5)
    sendline_and_wait_responses(sh, "git clone https://github.com/fretbuzz/mimir_v2.git", timeout=5)
    sh.sendline("pwd" )
    cur_cwd = sh.recvline(timeout=5)
    print "cur_cwd.split", cur_cwd.split(" ")
    cur_cwd = cur_cwd.split(" ")[-1].rstrip()
    print "cur_cwd", cur_cwd
    training_config_dir = cur_cwd + model_dir_with_data_name + '/'
    training_config_json = training_config_dir + model_config_file_name
    print "training_config_json", training_config_json
    eval_config_dir = cur_cwd + eval_dir_with_data_name + '/'
    eval_config_json = eval_config_dir + eval_config_file_name
    print "eval_config_json", eval_config_json

    eval_sed_mod_paths_cmd = "sed -i 's;" + eval_dir_with_data + ';' + eval_config_dir + ';\'' + ' ' + eval_config_json
    train_sed_mod_paths_cmd = "sed -i 's;" + model_dir + ';' + training_config_dir +';\'' + ' ' + training_config_json

    #print "eval_sed_mod_paths_cmd:::::", eval_sed_mod_paths_cmd
    #print "train_sed_mod_paths_cmd:::::", train_sed_mod_paths_cmd

    sendline_and_wait_responses(sh, eval_sed_mod_paths_cmd, timeout=15)
    sendline_and_wait_responses(sh, train_sed_mod_paths_cmd, timeout=15)

    sendline_and_wait_responses(sh, "cd ./mimir_v2/analysis_pipeline/", timeout=5)
    # sometimes you ONLY want to train the model... (so you can get through the sequential bottleneck faster)
    if training_config_json != eval_config_json:
        mimir_start_str =  "python mimir.py --training_config_json " + training_config_json + \
            " --eval_config_json " + eval_config_json
    else:
        mimir_start_str = "python mimir.py --training_config_json " + training_config_json
    print "mimir_start_str",mimir_start_str
    sendline_and_wait_responses(sh, mimir_start_str , timeout=240)

    # step (7) once completed, recover the end results (both model + eval dirs)
    # but first, reverse the sed that we did earlier...
    eval_sed_mod_paths_cmd = "sed -i 's;" + eval_config_dir + ';' + eval_dir_with_data + ';\'' + ' ' + eval_config_json
    train_sed_mod_paths_cmd = "sed -i 's;" + training_config_dir + ';' + model_dir +';\'' + ' ' + training_config_json
    sendline_and_wait_responses(sh, eval_sed_mod_paths_cmd, timeout=15)
    sendline_and_wait_responses(sh, train_sed_mod_paths_cmd, timeout=15)

    # the '../' is b/c I'm in too much of a hurry to just figure out what the above dir is...
    print "dont_retreive_eval",dont_retreive_eval
    print "dont_retreive_train", dont_retreive_train
    print "eval_config_dir",eval_config_dir, "eval_dir_with_data", eval_dir_with_data
    print "training_config_dir",training_config_dir, "model_dir",model_dir
    if not dont_retreive_eval:
        s.download(file_or_directory=eval_config_dir, local=eval_dir_with_data)
    if not dont_retreive_train:
        s.download(file_or_directory=training_config_dir, local=model_dir)

    # step (8) return some kinda relevant information (-- this'll be the eval cm's that are needed by the looper)
    ### the biggest question is how to get them -- simple. they must be saved by mimir in a text file, that we can then
    ### read and extract the values from...
    ## TODO ::: get self.rate_to_tg_to_cm ... don't really know where that is (physically) but prob easiest just to run and check


    # step (9) (not actually a part of this file) -- need to modify components of system
    # to be able to cope with being processed on another system -- mostly the file paths will
    # be wrong...
    return None

if __name__ == "__main__":
    remote_server_ip = "c240g5-110107.wisc.cloudlab.us"
    remote_server_key = "/Users/jseverin/Dropbox/cloudlab.pem"
    user =  "jsev"
    skip_install = True
    skip_upload = False
    dont_retreive_eval = False
    dont_retreive_train = False

    '''
    eval_data_dir = '/Volumes/exM/experimental_data/sockshop_info/sockshop_five_100/sockshop_five_100/'
    eval_analysis_config_file = eval_data_dir + 'sockshop_five_100_exp_proc.json'
    model_dir = "/Volumes/exM/experimental_data/sockshop_info/sockshop_five_100_mk2/sockshop_five_100_mk2/"
    model_analysis_config_file = model_dir + 'sockshop_five_100_mk2_exp_proc.json'
    '''
    eval_data_dir              =  '/Volumes/exM/experimental_data/sockshop_info/sockshop_four_100/sockshop_four_100/'
    eval_analysis_config_file  = 'sockshop_four_100.json'
    model_dir                  = '/Volumes/exM/experimental_data/sockshop_info/sockshop_four_100_mk2/sockshop_four_100_mk2/'
    model_analysis_config_file = 'sockshop_four_100_mk2.json'

    process_on_remote(remote_server_ip, remote_server_key, user, eval_data_dir, eval_analysis_config_file, model_dir,
                      model_analysis_config_file, skip_install=skip_install,
                      skip_upload=skip_upload, dont_retreive_eval=dont_retreive_eval,
                      dont_retreive_train=dont_retreive_train)