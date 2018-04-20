## this file is meant to bring the system to the next level of abstraction
## rather than just giving a single value (as in parameters.py), this file
## will specify a starting point, increment rate, and a stopping point
## this will allow the automatic looping of the experiment and creation
## of the desired graphs

### Note: this section remains unchanged from parameter.py b/c
### this only happens on sock shop setup, which only happens one

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

### END identical section

### Note: this section will need to change to simulate traffic that
### addresses the challenges our system will face

# number of background traffic locusts (i.e. generate the background
# traffic)
# Note: must be a string
num_background_locusts = "12"

# rate of spawning of background traffic locusts
rate_spawn_background_locusts = "1" # /sec

### END section that will need to be changed

### Note:  this is fine, though will probably need
### to be lengthened
# this is the total length of the experiment
# so the experiment keeps running for 
# desired_stop_time - desired_exfil_time
# after data exfiltration has taken place
desired_stop_time = 360

### END fine section


### Note: this section is the one that is going to need a bunch of 
### modifications

## The 'next-gen' in data exfiltration.
## keys are times, values are the bytes to exfiltrate (in one 5 sec segment)
## note: the bytes should probably be close to some linear combo of the big API calls (see exfil_data_V2)
## note: leave empty if you want never
## note: will show up in detection system 5 seconds after the given time, b/c exfiltration starts at the given
## time, so it won't be recorded by prometheus until 5 seconds later
exfils = {80 : 0, 180 : 0}

## this value specifies the increment(/decrement) for each of the exfils
## keys are times, values are increments, in bytes
exfil_increments = {80: 12500, 180: 12500}

## number of increments
## will run once using the 'exfils' values and then 
## will run the number of times given here, each with the increment applied
number_increments = 8

## number of repeat experiments
## this is the number of times to run each experiment
## (so number of experiments run =
## (number of increments) * (number of repeat experiments)
repeat_experiments = 3

## this is going to be the name of the experiment
## we are going to make a new directory in 'experimental_data'
## store all the files and graphs there, and maybe even a file
## that explains what each file is, b/c there is going to be 
## a lot of them.
experiment_name = "test_6_long"

## this is a list of the SENT svc_pair graphs that are displayed/saved
## Note: this list should be 1,2, or 4 items long
## anything else and it won't work
display_sent_svc_pair = [ ['front-end', 'user' ], ['front-end', 'orders' ], ['user', 'user-db'], ['front-end', 'carts']]

##this is a list of the RECEIVED svc_pair graphs that are displyed
display_rec_svc_pair = [ ['front-end', 'user' ], ['front-end', 'orders' ], ['user', 'user-db'], ['front-end', 'carts']]

# should graphs be displayed after every experiment
display_graphs = False

# how long into experiment before analyze?
start_analyze_time = 30
