import ast
import subprocess

def oddball_input_output(input_file, output_file, oddball_directory):
    name_to_number = convert_to_oddball_input(input_file, output_file, oddball_directory)


def convert_to_oddball_input(input_file, output_file, oddball_directory):
    # TODO: (1) need to but this into the oddball directory
    # TODO: (2) need to auto-run oddbal
    # TODO: (3) need to copy out the oddball results into the oddball info file (And delete the prev)
        # TODO: (3b) change back the names of the nodes and the add column titles
    file_name = input_file.split('/')[-1]


    # first, get a list of all nodes
    lines = None
    name_to_number = {} # b/c oddball takes numbers as input
    current_counter = 1
    output_lines = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    print lines[:5]
    for line in lines:
        print "line.split(\" \")",line.split(" ")
        node_one, node_two, pkts, bytes = line.split(" ")
        #print "attribs", attribs
        #attribs = ast.literal_eval(attribs)
        if node_one not in name_to_number:
            name_to_number[node_one] = current_counter
            current_counter += 1
        if node_two not in name_to_number:
            name_to_number[node_two] = current_counter
            current_counter += 1
        cur_weight = bytes# attribs['weight'] # TODO
        output_lines.append(str(name_to_number[node_one]) + ',' + str(name_to_number[node_two])
                            + ',' + str(cur_weight))
    # then, assign all nodes a #
    # then create a new output file
    with open(oddball_directory + file_name, 'w') as f:
        for line in output_lines:
            f.write(line)

    matlab_str = 'matlab -nodisplay -r "cd(\'/Users/jseverin/Downloads/oddball-lite\'); main(\'wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt\', 1);exit"'
    out = subprocess.check_output([matlab_str])
    print "out", out


    print "name_to_number", name_to_number
    return name_to_number

def convert_from_oddball_output(odd_ball_output_file, this_functions_output_file):
    # todo: look at the set of features.
    # split into the the categories by svc's
    # write a nice, easy to read csv file

    pass

if __name__== "__main__":
    #input_file = '/Volumes/exM2/experimental_data/wordpress_info/edgefiles/wordpress_six_rep_3__edgefile_dict.txt' ## TODO
    #line = None
    #with open(input_file)

    #                                                                                       wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt_features_all.txt
    input_file = "/Volumes/exM2/experimental_data/wordpress_info/edgefiles/pruned_edgefiles/wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt"
    output_file = "/Volumes/exM2/experimental_data/wordpress_info/edgefiles/oddball_edgefiles/wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt" ## TODO
    oddball_directory = '/Users/jseverin/Downloads/oddball-lite 2/'
    oddball_input_output(input_file, output_file, oddball_directory)
