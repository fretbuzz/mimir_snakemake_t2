import pandas as pd
import numpy as np

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
                #print "prev_vals_df", prev_vals_df
                #print prev_vals_df.ewm(span=20).mean()['vals'], prev_vals_df.ewm(span=20).std()
                #print type(prev_vals_df.ewm(span=20, min_periods=5).mean()['vals']), type(prev_vals_df.ewm(span=20, min_periods=5).std())
                #print prev_vals_df.ewm(span=20, min_periods=5).mean()['vals'][-1], prev_vals_df.ewm(span=20, min_periods=5).std()



                
                try:
                    ewma_mean = prev_vals_df.ewm(span=20, min_periods=5).mean()#['vals']#[-1]
                except:
                    ewma_mean = float('NaN')
                try:
                    ewma_stddev = prev_vals_df.ewm(span=20, min_periods=5).std()#['vals']#[-1]
                except:
                    ewma_stddev = float('NaN')
                print "EWMA_VALS", ewma_mean,ewma_stddev

                ewma_vectorized(prev_vals, alpha=0.25, offset=None, dtype=None, order='C', out=None)
                cur_val = G.get_edge_data(node_one, node_two, default={'weight':0})['weight']
                try:
                    anom_score = (cur_val - ewma_mean) / ewma_stddev
                    #print "anom_score", anom_score
                except:
                    anom_score = 0 # happens if ewma_stddev is zero...
                    #print "anom_score_was_NA"
                max_anom_score = max(max_anom_score, anom_score)
                old_dict[(node_one, node_two)].append(cur_val)
                old_dict[(node_one, node_two)] = old_dict[(node_one, node_two)][-20:]
    print "max_anom_score", max_anom_score
    return max_anom_score, old_dict

# literally copy-pasteed from:
#    https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    data = np.array(data, copy=False)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out