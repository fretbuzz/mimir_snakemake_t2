# this is the number of customer records that the sock shop's 
# database is loaded with during the setup phase
number_customer_records = 1000

# number of background traffic locusts (i.e. generate the background
# traffic)
# Note: must be a string
num_background_locusts = "12"

# rate of spawning of background traffic locusts
rate_spawn_background_locusts = "1" # /sec

# this is how long the experiment goes before data is startng to be
# exfiltrated (could be randomized later on??)
# note: make this number negative of you want exfiltration to NEVER happen
desired_exfil_time = 90 #60

# this is the total length of the experiment
# so the experiment keeps running for 
# desired_stop_time - desired_exfil_time
# after data exfiltration has taken place
desired_stop_time = 180

# this is the name of the pickle file where the recieved
# traffic matrixes will be kept
# note: this should always end in .pickle
# note: './experimental_data/' will be pre-pended to this
sent_matrix_location = "cumul_sent_matrix_total_bytes_exfil_at_90.pickle"

# this is the name of the pickle file where the sent
# traffic matrixes wil be kept
# note: this should always end in .pickle
# note: './experimental_data/' will be pre-pended to this
rec_matrix_location = "cumul_recieved_matrix_total_bytes_exfil_at_90.pickle"

## this is a list of the SENT svc_pair graphs that are displyed
## Note: this list should be 1,2, or 4 items long
## anything else and it won't work
display_sent_svc_pair = [ ['front-end', 'user' ], ['front-end', 'orders' ], ['user', 'user-db'], ['front-end', 'carts']]

##this is a list of the RECEIVED svc_pair graphs that are displyed
display_rec_svc_pair = [ ['front-end', 'user' ], ['front-end', 'orders' ], ['user', 'user-db'], ['front-end', 'carts']]

#### TODO: How much data to exfiltrate? Or is that already handled
#### by the loading-the-database parameter (cause it steals all of it)a
