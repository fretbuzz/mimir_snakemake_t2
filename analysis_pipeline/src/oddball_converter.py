import ast
import subprocess

path_to_matlab = '/Applications/MATLAB_R2017a.app/bin/matlab'

def oddball_input_output(input_file, output_file, oddball_directory):
    name_to_number = convert_to_oddball_input(input_file, output_file, oddball_directory)


def convert_to_oddball_input(input_file, output_file, oddball_directory):
    # TODO: (1) need to but this into the oddball directory [DONE]
    # TODO: (2) need to auto-run oddbal [DONE]
    # TODO: (3) need to copy out the oddball results into the oddball info file (And delete the prev)
        # TODO: (3b) change back the names of the nodes and the add column titles
    ## TODO: okay, might need to do part (3) still... but I am way more skeptical of how useful it is...
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

    matlab_str = path_to_matlab + ' -nodisplay -r "cd(\'' + oddball_directory + '\'); main(\'' + file_name + '\', 1);exit"'
    print "matlab_str", matlab_str
    out = subprocess.check_output([matlab_str], shell=True)
    print "out", out

    print  "first_name_to_number", name_to_number
    return name_to_number

def convert_from_oddball_output(odd_ball_output_file, this_functions_output_file, name_to_number):
    # todo: look at the set of features.
    # split into the the categories by svc's
    # write a nice, easy to read csv file
    number_to_name = {}
    for name,number in name_to_number.iteritems():
        number_to_name[number] = name

    file_name = input_file.split('/')[-1]
    #oddball_out = oddball_directory + file_name + '_features_all.txt'
    oddball_lines = None
    with open(odd_ball_output_file, 'r') as f:
        oddball_lines = f.readlines()

    names_of_columns = 'NodeName  egonetN egonetE egonetW     egonetInD egonetOutD egonetInW     egonetOutW    egoInD   egoOutD  egoInW egoOutW   egonetMaxW egoMaxInW egoMaxOutW numneighdeg1'
    print "this_functions_output_file",this_functions_output_file
    with open(this_functions_output_file, 'w') as f:
        f.write(names_of_columns + '\n')
        for line_counter,line in enumerate(oddball_lines):
            f.write(number_to_name[line_counter+1] + '\t' + line)

    print "oddball_lines", oddball_lines

    print "name_to_number", name_to_number

if __name__== "__main__":
    #input_file = '/Volumes/exM2/experimental_data/wordpress_info/edgefiles/wordpress_six_rep_3__edgefile_dict.txt' ## TODO
    #line = None
    #with open(input_file)

    #                                                                                       wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt_features_all.txt
    input_file = "/Volumes/exM2/experimental_data/wordpress_info/edgefiles/pruned_edgefiles/wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt"
    output_file = "/Volumes/exM2/experimental_data/wordpress_info/edgefiles/oddball_edgefiles/wordpress_six_rep_3_default_bridge_0any_split_00000_20180927175124_edges.txt" ## TODO
    oddball_directory = '/Users/jseverin/Downloads/oddball-lite 2/'
    file_name = input_file.split('/')[-1]
    oddball_output_file = oddball_directory + file_name + '_features_all.txt'
    total_output_file = '/Volumes/exM2/experimental_data/wordpress_info/oddball_info/' + file_name

    name_to_number = convert_to_oddball_input(input_file, output_file, oddball_directory)
    print "name_to_number_two", name_to_number
    convert_from_oddball_output(oddball_output_file, total_output_file, name_to_number)
