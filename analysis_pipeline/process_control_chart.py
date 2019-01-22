import pandas as pd

def ewma_control_chart_max_val(G, old_dict):
    nodes = G.nodes()
    max_anom_score = 0
    for node_one in nodes:
        for node_two in nodes:
            if node_one != node_two:
                try:
                    prev_vals = old_dict[(node_one, node_two)]
                except:
                    old_dict[(node_one, node_two)] = []
                    prev_vals = []
                prev_vals_df = pd.DataFrame({'vals': prev_vals})
                print "prev_vals_df", prev_vals_df
                print prev_vals_df.ewm(span=20).mean()['vals'], prev_vals_df.ewm(span=20).std()
                print type(prev_vals_df.ewm(span=20, min_periods=5).mean()['vals']), type(prev_vals_df.ewm(span=20, min_periods=5).std())
                #print prev_vals_df.ewm(span=20, min_periods=5).mean()['vals'][-1], prev_vals_df.ewm(span=20, min_periods=5).std()
                try:
                    ewma_mean = prev_vals_df.ewm(span=20, min_periods=5).mean()['vals'][-1]
                except:
                    ewma_mean = float('NaN')
                try:
                    ewma_stddev = prev_vals_df.ewm(span=20, min_periods=5).std()['vals'][-1]
                except:
                    ewma_stddev = float('NaN')
                print ewma_mean,ewma_stddev

                cur_val = G.get_edge_data(node_one, node_two, default={'weight':0})['weight']
                try:
                    anom_score = (cur_val - ewma_mean) / ewma_stddev
                except:
                    anom_score = 0 # happens if ewma_stddev is zero...
                max_anom_score = max(max_anom_score, anom_score)
                old_dict[(node_one, node_two)].append(cur_val)
                old_dict[(node_one, node_two)] = old_dict[(node_one, node_two)][-20:]
    return max_anom_score, old_dict