import os

def main(exp_parent_directory):
    # step 1: I think we should pass this function the over-arching directory that contains the experimental directory
    ## we need to find all the .log files within this directory
    log_files_from_exps = find_logfile_paths(exp_parent_directory)
    # step 2: for each file found above, read it in and call the sanity-checking function
    print "log_files_from_exps",log_files_from_exps
    logfiles_with_problems = []
    for logfile_name in log_files_from_exps:
        with open(logfile_name, 'r') as f:
            logfile_contents = f.read()

        # step 3: write the sanity-checking function: look for just two things at the moment:
        if is_there_problem_in_logfile(logfile_contents):
            logfiles_with_problems.append(logfile_name)

    # step 4: output a table(? list?) showing which experiments had which problems
    print "logfiles_with_problems"
    for logfile_with_problem in logfiles_with_problems:
        print "  ", logfile_with_problem

def is_there_problem_in_logfile(logfile_contents):
    # First thing to find: uncaught exceptions
    if 'uncaught exception' in logfile_contents:
        return True
    # Second thing to find: pulling the pcap in the middle of an experiment (so strange...)
    # TODO

    return False

def find_logfile_paths(exp_parent_directory):
    log_files = []
    for subdir, dirs, files in os.walk(exp_parent_directory):
        results_directory = subdir + '/results/'
        print "results_directory", results_directory

        log_file = ""
        for subdir, dirs, files in os.walk(exp_parent_directory):
            print "files in results directory", files
            for file in files:
                if '.log' in file:
                    log_file = file
                    break
        if log_file != '':
            log_files.append(log_file)
    return log_files

if __name__ == "__main__":
    logfile_name = "/Users/jseverin/Documents/sockshop_scale_test_trial" # TODO<- put it here...
    main(logfile_name)