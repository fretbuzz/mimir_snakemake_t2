import docker
import networkx as nx
import os
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def run_mulval():
    client = docker.from_env()
    client.containers.list()
    cwd = os.getcwd()
    print "cwd", cwd
    print "mulval_related_dir", cwd + '/mulval_inouts'
    # note: going to start container running everytime (and later on will detelete it...)
    container = client.containers.run("risksense/mulval", detach=True,
                                      volumes={cwd + '/mulval_inouts': {'bind': '/mnt/vol2', 'mode': 'rw'}})
    print "cwd", cwd

    commands = [ ['/root/mulval/utils/graph_gen.sh', '/mnt/vol2/test_mulval_input.P'],
                 ['mv', '/root/mulval/utils/xsb_log.txt', '/mnt/vol2/xsb_log.txt'],
                 ['mv', '/root/mulval/utils/AttackGraph.txt', '/mnt/vol2/AttackGraph.txt']]
    for command in commands:
        out = container.exec_run(command, workdir='/root/mulval/utils/')
        print out #out.output, out

    print client.containers.list()

    print container.stop()
    print container.remove()
    #print client.containers.prune(filters={'id': container.id})


def prepare_mulval_input(ms_s):
    # this is going to need to take the k8s input files and then convert to the input format for mulval
    ### TODO: okay, I think that this is the next step...
    ### Stuff to include:
    ###    internet <-> anything it wants, anything it wants
    ###    pods  -> send anywhere, anything
    ###    vips -> get set of three rules, heavily dictating what they accept and what they don't
    ###    + each pod can get the same vulnerability (so 2 lines + shared)
    ###    + each vip gets its own vulnerability (so 3 lines...)

    ## so we can break this down into two steps...
    ## (1) acquire all the config-related information
    ## (2) convert it into the mulval-compatible form...

    ## okay, well let's obviously start w/ (1)... can we leverage existing parsing
    ## capabilities to get it done?? looks like kinda, but not completely, b/c it puts everything
    ## into the 'mapping' dict, instead of keeping everything seperate...
    ## just us the ms (list of microservices) to filter the mappings and that'll be fine...
    ## so next step is wiring it in... comment out other parts afterr

    # step 1a: get list of services + pods... wait no, that's wrong... we only care about ms_s...
    # the mapping to the concrete pods happens in postprocessing... so we can create the model
    # declarivitevly via the following logic...
    svcs = []
    for ms in ms_s:
        svcs.append( (ms + '_pod', ms + '_vip') )

    # step 2. let's generate the corresponding test_mulval_input.P
    # step 2.a. the internet (can talk to/from anything)
    lines = ['hacl(internet, _, _, _).', 'hacl(_, internet, _, _).']

    # step 2.b. the hacl's ; do simulteanously for VIPs and Pods
    # recall that pods can do whatever they want, vips are more limited (obvi)
    for svc in svcs:
        # pod is easy
        lines.append( 'hacl(' + svc +', _, _, _).' )
        # VIP is harder
        ## TODO: VIP
        ## NOTE: I need port information!!! This will require reworking the mapping functions!!!!
        #### TODO: Immediate next steps
        ####   (a) modify mapping to get ports + feed them into mulval component
        ####   (b) incorporate logic into this
        ####   (c) finish input preparer by modifying new_old_test_mulval_input w/ application-info
        ####   (d) finish the output processor (might be a bit of work...)
        #### would be nice if I could get through c today

        pass # let's do this!!

    # step 2.c. then do the vulneabiltiy stuff (should be very 'boilerplate')


    with open('./mulval_inouts/auto_test_mulval_input.P', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    return # do I want to return anything??

def post_process_mulval_result():
    # well, not in accordance with the architecture document, if we're putting orchestrator logic into the
    # mulval component, then we'd need to include the 'chaining' together of various steps

    graph = open('./mulval_inouts/AttackGraph.txt', 'r')
    lines = graph.readlines()
    hacls = []
    for line in lines:
        if 'hacl' in line:
            hacls.append(line.rstrip().lstrip())
    propogation_graph_edges = []
    for hacl in hacls:
        only_hacl =  hacl.split('-', 1)[1]
        only_relevant = only_hacl.split('(')[1].split(')')[0]
        #print only_hacl, only_relevant.split(',')#onfly_hacl.split('('), only_hacl.split(',')
        propogation_graph_edges.append(only_relevant.split(','))

    # okay, for now we should use the simplest visualization process possible... networkx it is then...
    G = nx.DiGraph() # this'll be the propogation graph...
    # okay, so it looks like it is going to be some kinda loop....
    # looks like G.add_edge(NODE1, NODE2, weight=WEIGHT) is the way to do this...
    # so all we gotta do is a construct a set of tuples from the hacls... piece of cake...
    for edge in propogation_graph_edges:
        G.add_edge(edge[0], edge[1], proto = edge[2], port = edge[3])
        #print edge

    for edge in G.edges():
        print "edge", edge

    #nx.draw(G)
    #plt.draw()

    pos = graphviz_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, arrows=True)

    plt.show()
    print "has the graph been drawn???? yes, yes they have"

def generate_synthetic_attack_templates(mapping, ms_s):
    prepare_mulval_input(ms_s)
    run_mulval()
    post_process_mulval_result()

if __name__ == "__main__":
    generate_synthetic_attack_templates([], [])