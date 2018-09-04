# note: this file has hardcoded logic but works. Should generalize at some point.
import csv
import matplotlib.pyplot as plt
import numpy as np
import time

# NOTE: IS THIS REALLY CALCULATING WHAT I WANT IT TO THOUGH? WHAT I WANT IS TO CALC
# THE

a_list = []
b_list = []
conn_to_weight = {}

for i in range(0, 1198, 1):
     with open('/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_full_scaled__' + '%.2f' % float(i) + '_1.00.txt',  'r') as csvfile:
        csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        a,b = 0,0
        for row in csvread:
            if 'default' in row[0] and 'default' in row[1]:
                if (row[0], row[1]) in conn_to_weight:
                    conn_to_weight[row[0], row[1]].append( int(row[2]) )
                else:
                    conn_to_weight[row[0], row[1]] = [int(row[2])]

                #print row
            '''
            if 'my-release-pxc' in row[0] and 'wordpress' in row[1] and 'VIP' not in row[0]:
                # b is the amt of bytes from the DB containers to the wordpress containers
                b += int(row[2])
            if 'pxc_VIP' in row[0] and 'wordpress' in row[1]:
                # b is the amt of bytes from the DB VIP to the wordpress containers 
                a += int(row[2])
            '''
        print i, a,b, "    ", b-a ##b-a=data from DB containers to wordpress containers that doesn't use the VIP
        a_list.append(a)
        b_list.append(b)
real_conn_to_quarts = {}
real_conn_to_coefs = {}
classes = ['wordpress', 'pxc']
class_to_instances = {}
for cl in classes:
    for name in conn_to_weight.keys():
        name = name[0]
        if cl in name:
            if cl in class_to_instances:
                class_to_instances[cl].append( name )
                class_to_instances[cl] = list(set(class_to_instances[cl]))
            else:
                class_to_instances[cl] = [name]

print "class_to_instances", class_to_instances
#print conn_to_weight['default/wwwppp-wordpress-7b79f985d-kngnr', 'default/my-release-pxc-1']
#time.sleep(20)

class_to_maxs_minus_mins = {}
for i in range(0,268):
    for cl1 in classes:
        for cl2 in classes:
            vals = []
            for inst in class_to_instances[cl1]:
                for inst2 in class_to_instances[cl2]:
                    try:
                        #print inst, inst2, conn_to_weight[inst, inst2][i], 'end'
                        vals.append( int(conn_to_weight[inst, inst2][i]))
                    except:
                        continue
            print cl1, 'to', cl2, 'at', i, "vals", len(vals), vals
            if vals:
                if (cl1, cl2) in class_to_maxs_minus_mins:
                    class_to_maxs_minus_mins[cl1, cl2].append(  max(vals) - min(vals)  )
                else:
                    class_to_maxs_minus_mins[cl1, cl2] = [  max(vals) - min(vals)   ]
print class_to_maxs_minus_mins

print "hi"
for name,val in conn_to_weight.iteritems():
    print name, sum(val)
    #print name,val
    '''
    val2 = val
    conn_to_quarts = [] # note: name is misleading
    conn_to_coefs = [] # note: name is misleading
    for i in range(1,len(val2)):
        val = val2[:i]
        #print name,val
        # okay, cool, let's calc some values
        # (1) quartile coefficient of dispersion (Q3 - Q1) / (Q3 + Q1)
        val = [int(i) for i in val]
        Q3, Q1 = np.percentile(val, [75, 25])
        quartile_coef_of_disp = (Q3 - Q1) / (Q3 + Q1)
        conn_to_quarts.append(quartile_coef_of_disp)
        #print "quartile_coef_of_disp", quartile_coef_of_disp
        # (2) coefficient fo variation
        # (standard deviation) / (mean)
        coef_of_var = np.std(val) / np.mean(val)
        conn_to_coefs.append(coef_of_var)
        #print "coef_of_var", coef_of_var
        # (3) mean absolute deviaton from the mean
    print "quartile_coef_of_disp", conn_to_quarts
    print "coef_of_var", conn_to_coefs
    real_conn_to_quarts[name] = conn_to_quarts
    real_conn_to_coefs[name] = conn_to_coefs
    '''

for name, val in real_conn_to_quarts.iteritems():
    plt.figure(1)
    plt.hist(val)
    plt.title(name)
    plt.ylabel('# of occurs')
    plt.xlabel('quartile coefficient of dispersion')
    plt.figure(2)
    plt.hist(real_conn_to_coefs[name])
    plt.title(name)
    plt.ylabel('# of occurs')
    plt.xlabel('coefficient of variation')
    plt.show()

x_vals = [i for i in range(0,268)]
print len(x_vals), len(a_list)
'''
plt.scatter(x_vals[0:99], [b - a for a,b in zip(a_list, b_list)][0:99], color='black')
plt.scatter(x_vals[99:149], [b - a for a,b in zip(a_list, b_list)][99:149], color='r')
plt.scatter(x_vals[150:], [b - a for a,b in zip(a_list, b_list)][150:], color='black')
plt.title('wordpress_four DB_containers->WP_containers - DB_VIP->WP_containers')
plt.ylabel('inter-service data that doesn\'t go through VIP (bytes)')
plt.xlabel('time (sec)')
plt.savefig('./wordpress_four_VIP_invariant.png')
plt.show()
'''
