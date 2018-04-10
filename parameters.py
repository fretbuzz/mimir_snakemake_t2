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
desired_exfil_time = 60

# this is the total length of the experiment
# so the experiment keeps running for 
# desired_stop_time - desired_exfil_time
# after data exfiltration has taken place
desired_stop_time = 100

#### TODO: How much data to exfiltrate? Or is that already handled
#### by the loading-the-database parameter (cause it steals all of it)a
