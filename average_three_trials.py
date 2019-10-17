
import json
from remote_experiment_runner import run_experiment
from analysis_pipeline.generate_paper_graphs import run_looper, generate_graphs

def main(config_files):
    #print "okay, here we go"
    '''I was going to take a single remote_experiment_config file and then autogenerate the set for a trial... but this
    is too much automation, without a good reason. You loose a lot of flexibility to save a teeny bit of time. So, I'm just
    going to have it take a set of config files instead...'''
    '''
    base_config_file_core = base_config_file[-5:]
    print "base_config_file_core", base_config_file_core

    with open(base_config_file, "r") as read_file:
        base_config = json.load(read_file)

    dir_with_trial_configs = "/".join( base_config_file.split('/')[:-1] ) + '/' + 'trials/'
    base_corresponding_local_dir = base_config['corresponding_local_directory']
    for counter, machine_ip in enumerate(machine_ips):
        base_config['corresponding_local_directory']  = None # TODO TODO TODO TODO TODO

        # TODO: write this to file
        # add something onto the end of dir_with_trial_configs and then just use the json capabilities to write it...
    '''

    list_eval_configs_to_xvals_and_cm = []

    for counter, config_file in enumerate(config_files):
        corresponding_multi_experiment_config_file = None
        with open(config_file, "r") as read_file:
            config = json.load(read_file)
            # corresponding_local_directory
            #corresponding_local_directory = config['corresponding_local_directory']
            corresponding_multi_experiment_config_file = None # TODO
        return_local_data_only_p = config['return_local_data_only']

        # todo: need multiprocess for multithreading...
        if not return_local_data_only_p:
            run_experiment(config_file, only_retrieve=False) # if you want to use only_retrieve, call that func directly...

        # TODO: okay, this next part is alot harder than I initially thought b/c the multi_experiment_config_file (or something
        # like that) is embedded in the e2e script...
        ## !!!! I need the remote looper to move the multi_experment_config_file into the outer part of the nested
        # directory... that way I can grab it with the corresponding_local_directory attribute...]!!!
        # put this up there ^^^
        # (I'll also need to adjust the relative paths of some components in th  file)

        # okay, once we get here, let's assume that generate_paper_graphs can grab the values from cache...
        # todo: right here, we need to get the DFs for all the individual sets of exps... and then we wan to avg these results...
        eval_configs_to_xvals, exfil_rate, evalconfigs_to_cm, model_config_file = \
            run_looper(corresponding_multi_experiment_config_file, update_config=False, use_remote=False,
                       only_finished_p=True, live_p=False)
        # todo: accumulate the dfs somewhere...
        list_eval_configs_to_xvals_and_cm.append( (eval_configs_to_xvals, evalconfigs_to_cm) )

    # todo: now take the average of the DFs
    ## TODO: okay, so what goes into this section??: well, we need to ensure that the values are
    ### NOTE: realistically, I'm going to need the part above running smoothly and having it be easy to reach
    # this part, so I can use the debugger to see what exactly is going on in the heavily-nested evalconfigs_to_cm

    # todo: now call run_looper again,but with this new averaged DF...
    # most of these values can be
    #generate_graphs(eval_configs_to_xvals, exfil_rate, evalconfigs_to_cm, timegran, type_of_graph, graph_name, xlabel,
    #                model_config_file, no_tsl, model_xval)

    print "okay, more stuff to do here..."


    pass

if __name__=="__main__":
    print "RUNNING"
    config_files = ['./remote_experiment_configs/trials/sockshop_scale_trial_1_rep1.json',
                    './remote_experiment_configs/trials/sockshop_scale_trial_1_rep2.json',
                    './remote_experiment_configs/trials/sockshop_scale_trial_1_rep3.json']
    #machine_ips = ["c240g5-110119.wisc.cloudlab.us","c240g5-110129.wisc.cloudlab.us","c240g5-110139.wisc.cloudlab.us",]
    main(config_files)