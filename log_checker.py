import os, time

def main(exp_parent_directory):
    # step 1: I think we should pass this function the over-arching directory that contains the experimental directory
    ## we need to find all the .log files within this directory
    log_files_from_exps, deploy_log_files = find_logfile_paths(exp_parent_directory)
    # step 2: for each file found above, read it in and call the sanity-checking function
    print "---------"
    print "log_files_from_exps",log_files_from_exps
    print "deploy_log_files", deploy_log_files
    logfiles_with_problems = []
    deploy_logfiles_with_problems = []
    for counter, logfile_name in enumerate(log_files_from_exps):
        with open(logfile_name, 'r') as f:
            logfile_contents = f.read()

        # step 3: write the sanity-checking function: look for just two things at the moment:
        if is_there_problem_in_logfile(logfile_contents):
            logfiles_with_problems.append(logfile_name)

    for counter, logfile_name in enumerate(deploy_log_files):
        with open(logfile_name, 'r') as g:
            deploy_logfile_contents = g.read()

        #print logfile_name, is_there_problem_in_deploy_log(deploy_logfile_contents)

        is_problem, problem_lines = is_there_problem_in_deploy_log(deploy_logfile_contents)
        if is_problem:
            deploy_logfiles_with_problems.append((logfile_name, problem_lines))

    #print "deploy_logfiles_with_problems", deploy_logfiles_with_problems

    # step 4: output a table(? list?) showing which experiments had which problems
    print "####################"
    print "logfiles_with_problems"
    for logfile_with_problem in logfiles_with_problems:
        print "  ", logfile_with_problem

    print "deploy_logfiles_with_problems"
    for deploy_logfile_with_problems in deploy_logfiles_with_problems:
        print "deploy_logfile_with_problem:", deploy_logfile_with_problems[0]
        print "problems here: ", deploy_logfile_with_problems[1]

    print "####################"
    if len(logfiles_with_problems) + len(deploy_logfiles_with_problems) == 0:
        print "Congratulations, no problems in the collected data!"
        return True
    else:
        print "Sorry, there is one or more problems in the collected data"
        return False

def is_there_problem_in_logfile(logfile_contents):
    # First thing to find: uncaught exceptions
    if 'uncaught exception' in logfile_contents:
        return True
    # Second thing to find: pulling the pcap in the middle of an experiment (so strange...)
    # TODO
    # Third thing to find: make sure that netshoot started succesfully...
    #if 'Welcome to Netshoot!' not in logfile_contents:
    #    return True
    # not sure if needed/wanted...

    return False

def is_there_problem_in_deploy_log(deploy_log_contents):
    deploy_log_contents = deploy_log_contents.split('\n')
    # Look to see if there is 0/N pods ready for a particular deployment at each log point...
    deploy_name_ready_desired = []
    for counter, line in enumerate(deploy_log_contents):
        line = line.strip()
        #print 'z', line
        #if len(line) > 0:
        #    print 'k', len(line) * line[0]
        #    print not (line == len(line) * line[0])
        if line != "" and not (line == len(line) * line[0]):
            #print "FFDF"
            line_comp = line.split()  # .split(" ")
            #print "line_comp", line_comp
            try:
                pods_ready = int(line_comp[2].split('/')[0])
                pods_desired = int(line_comp[2].split('/')[1])
                deploy_name = line_comp[1]
                #print deploy_name, pods_ready, '/', pods_desired
                deploy_name_ready_desired.append( (deploy_name, pods_ready, pods_desired, counter) )
            except:
                pass

    #print "deploy_name_ready_desired",deploy_name_ready_desired
    #time.sleep(300)

    problem_lines = []
    for dnrd in deploy_name_ready_desired:
        if dnrd[1] == 0:
            problem_lines.append(dnrd[3])

    if len(problem_lines) != 0:
        is_problem = True
    else:
        is_problem = False

    return is_problem, problem_lines

def find_logfile_paths(exp_parent_directory):
    log_files = []
    deploy_log_files = []
    for subdir, dirs, files in os.walk(exp_parent_directory):
        #results_directory = subdir + '/results/'
        #print "results_directory", results_directory

        log_file = ""
        deploy_log_file = ""
        for subdir, dirs, files in os.walk(exp_parent_directory):
            #print "current_subdir", subdir
            #print "files in results directory", files

            for file in files:
                if '.log' in file:
                    log_file = subdir + '/' + file

                # also want to find the deploy config files (look for _deploy_config in filename...)
                if '_deploy_config_.txt' in file:
                    deploy_log_file = subdir + '/' + file

            if log_file != '':
                log_files.append(log_file)
                deploy_log_files.append(deploy_log_file)

    log_files = list(set(log_files))
    deploy_log_files = list(set(deploy_log_files))
    return log_files, deploy_log_files

if __name__ == "__main__":
    exp_directory = "/Users/jseverin/Documents/sockshop_scale_test_trial" # TODO<- put it here...
    main(exp_directory)