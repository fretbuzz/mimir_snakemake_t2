# the three following values determine the number of customer records 
# that the sock shop's database is loaded with during the setup phase
# for more information on why this is necessary, see GitHub issue #25

# these users will be registered, given an address, and given 
# CC information (so that they can be used to buy things in background
# simulation)
# note: multiply this number by number of desired records by 4
number_full_customer_records = "620"

# these users will be registered and given an address
# note: multiply the number of desired records by 3
number_half_customer_records = "900"

# these users will just be registed
# note: multiple the number of desired records by 1 (not a type)
number_quarter_customer_records = "1000"

# number of background traffic locusts (i.e. generate the background
# traffic)
# Note: must be a string
num_background_locusts = "12"

# rate of spawning of background traffic locusts
rate_spawn_background_locusts = "1" # /sec

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

## The 'next-gen' in data exfiltration.
## keys are times, values are the bytes to exfiltrate (in one 5 sec segment)
## note: the bytes should probably be close to some linear combo of the big API calls (see exfil_data_V2)
## note: leave empty if you want never
exfils = {40 : 50000, 90 : 100000}

#### TODO: How much data to exfiltrate? Or is that already handled
#### by the loading-the-database parameter (cause it steals all of it)a
