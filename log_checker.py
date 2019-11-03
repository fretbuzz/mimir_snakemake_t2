import os

def main(exp_parent_directory):
    # step 1: I think we should pass this function the over-arching directory that contains the experimental directory
    ## we need to find all the .log files within this directory
    log_files_from_exps, deploy_log_files = find_logfile_paths(exp_parent_directory)
    # step 2: for each file found above, read it in and call the sanity-checking function
    print "log_files_from_exps",log_files_from_exps
    logfiles_with_problems = []
    deploy_logfiles_with_problems = []
    for counter, logfile_name in enumerate(log_files_from_exps):
        deploy_logfile_name = deploy_log_files[counter]
        with open(logfile_name, 'r') as f:
            logfile_contents = f.read()
        with open(deploy_logfile_name, 'r') as g:
            deploy_logfile_contents = g.read()

        # step 3: write the sanity-checking function: look for just two things at the moment:
        if is_there_problem_in_logfile(logfile_contents):
            logfiles_with_problems.append(logfile_name)

        if is_there_problem_in_deploy_log(deploy_logfile_contents):
            deploy_logfiles_with_problems.append(deploy_logfile_name)

    # step 4: output a table(? list?) showing which experiments had which problems
    print "logfiles_with_problems"
    for logfile_with_problem in logfiles_with_problems:
        print "  ", logfile_with_problem

    print "deploy_logfiles_with_problems"
    for deploy_logfile_with_problems in deploy_logfiles_with_problems:
        print " ", deploy_logfile_with_problems

def is_there_problem_in_logfile(logfile_contents):
    # First thing to find: uncaught exceptions
    if 'uncaught exception' in logfile_contents:
        return True
    # Second thing to find: pulling the pcap in the middle of an experiment (so strange...)
    # TODO
    # Third thing to find: make sure that netshoot started succesfully...
    if 'Welcome to Netshoot!' not in logfile_contents:
        return True

    return False

def is_there_problem_in_deploy_log(deploy_log_contents):
    # Look to see if there is 0/N pods ready for a particular deployment at each log point...
    deploy_name_ready_desired = []
    for line in deploy_log_contents:
        line = line.strip()
        if line != "" and not (line == len(line) * line[0]):
            line_comp = line.split()  # .split(" ")
            try:
                pods_ready = int(line_comp[2].split('/')[0])
                pods_desired = int(line_comp[2].split('/')[1])
                deploy_name = line_comp[1]
                print deploy_name, pods_ready, '/', pods_desired
                deploy_name_ready_desired.append( (deploy_name, pods_ready, pods_desired) )
            except:
                pass

    for dnrd in deploy_name_ready_desired:
        if dnrd[1] == 0:
            return True
    return False

def find_logfile_paths(exp_parent_directory):
    log_files = []
    deploy_log_files = []
    for subdir, dirs, files in os.walk(exp_parent_directory):
        results_directory = subdir + '/results/'
        print "results_directory", results_directory

        log_file = ""
        deploy_log_file = ""
        for subdir, dirs, files in os.walk(exp_parent_directory):
            print "files in results directory", files
            for file in files:
                if subdir + '.log' in file:
                    log_file = file
                    break

                # also want to find the deploy config files (look for _deploy_config in filename...)
                if subdir + '_pod_config_.txt' in file:
                    deploy_log_file = file
                    break
        if log_file != '':
            log_files.append(log_file)
            deploy_log_files.append(deploy_log_file)
    return log_files, deploy_log_files

if __name__ == "__main__":
    exp_directory = "/Users/jseverin/Documents/sockshop_scale_test_trial" # TODO<- put it here...
    main(exp_directory)