import csv
import matplotlib.pyplot as plt

a_list = []
b_list = []
list_of_per_container = []
for i in range(0, 570, 30):
     #print "time:", str(i)
     per_container = {}
     with open('wordpress_six_take_2__' + '%.2f' % float(i) + '_30.00.txt',  'r') as csvfile:
        csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        a,b = 0,0
        for row in csvread:
            #if ('my-release-pxc' in row[0] or 'wordpress' in row[0]) and '10.1.0.2' in row[1] and 'VIP' not in row[0]:
	    '''if '10.1.0.2' in row[1]: # and '192.168' not in row[0]: # and '10.1' not in row[0]:# and 'VIP' not in row[0]:
		print row 
                # b is the amt of bytes from the DB containers to the wordpress containers
                b += int(row[2])
                if row[1] in per_container.keys():
			per_container[row[1]] += int(row[2])
                else:
			per_container[row[1]] = int(row[2])
            '''
	    if '10.1.0.2' in row[0] and '192.168.99.1' == row[1]:
                # b is the amt of bytes from the DB VIP to the wordpress containers
		#print row
                a += int(row[2])
            if 'kube-dns_VIP' in row[1]:
		print row
        	b += int(row[2])
	if b != 0:
		ratio = float(a) / float(b)
	else:
		ratio = 0	

	print i, a,b, "    ", ratio #float(a)/float(b) ##b-a=data from DB containers to wordpress containers that doesn't use the VIP
        a_list.append(a)
        b_list.append(b)
	list_of_per_container.append(per_container)
x_vals = [i for i in range(0,268)]
print len(x_vals), len(a_list)
print a_list
print b_list
print list_of_per_container


#'''
plt.scatter(x_vals[0:99], [a for a,b in zip(a_list, b_list)][0:249], color='black')
plt.scatter(x_vals[99:149], [a for a,b in zip(a_list, b_list)][248:299], color='r')
plt.scatter(x_vals[150:], [a for a,b in zip(a_list, b_list)][299:], color='black')
plt.title('wordpress_four DB_containers->WP_containers - DB_VIP->WP_containers')
plt.ylabel('inter-service data that doesn\'t go through VIP (bytes)')
plt.xlabel('time (sec)')
plt.savefig('./wordpress_four_VIP_invariant.png')
plt.show()
#'''

