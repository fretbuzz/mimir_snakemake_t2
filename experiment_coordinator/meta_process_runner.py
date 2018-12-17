# this file exists b/c the garbage collector in python is wierd
# so if we start the processing with a subprocess command, we know that the memory can
# be freed when the processor ends, and we can therefore run several processing sessions in a row
# without fear of it being killed due to lack of memory

import subprocess

'''
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress6_rep4"])
print out
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress7_rep3"])
print out
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress8"])
print out

'''
# for use later...
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress6_rep3"])
print out
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress7_rep2"])
print out
#out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress6_rep2"])
#print out
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress8_rep_2"])
print out
out = subprocess.check_output(["python", "run_analysis_pipeline_recipes.py", "process_wordpress8_rep_3"])
print out
#'''