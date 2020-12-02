from utils import scale_fitter_no_grid, utils
import numpy as np
import itertools 
#from joblib import Parallel, delayed
from collections import Counter, defaultdict
import math
#from importlib import reload
from math import radians, cos, sin, asin, sqrt
#import multiprocessing
#from tqdm import tqdm_notebook as tqdm
#import pandas as pd
#reload(scale_fitter_no_grid)
import scipy
import geopy
import geopy.distance


def scales_generator(trace, size, stop_coords, self_transitions=False):
    '''
    Generate a synthetic trace starting from a sequence of locations with the corresponding scale structure
    
    Input
    -----
    trace: list of lists (e.g.[ [1,2,3] , [1,2,1], ...., [2,1,1]])
    size: len of the synthetic trace
    
    
    Output
    ------
    synthetic_trace: list of lists (e.g.[ [1,2,3] , [1,2,1], ...., [2,1,1]])
    
    '''
    
    traces_len = int(size)
    n_scales = len(trace[0])
    trace = list(trace)
    
    #Create the source_target_list and compute the parameters of the model
    source_target = np.stack([trace[:-1], trace[1:]], axis=1)
    (proba_dist, proba_dist_counts), cell_attractiveness, cell_p_change, alphas = scale_fitter_no_grid.compute_likelihood(source_target, return_all_values=True)

    #Create nested dictionary
    nested_dictionary = []
    items = sorted(cell_attractiveness.items(), key = lambda x:len(x[0]))
    for group1 in itertools.groupby(items,lambda x:len(x[0])):
        scale = group1[0]
        new_group = sorted(list(group1[1]), key= lambda x:x[0][:scale-1])
        new_dict = dict()
        for group2 in itertools.groupby(new_group,lambda x:x[0][:scale-1]):
            new_dict[group2[0]] = dict(group2[1])
        nested_dictionary.append(new_dict)

    #Initialize synthetic trace
    L = tuple(trace[-1]) #current cell
    synthetic_series = [L[-1]] #sequence of cells
    scale_change = cell_p_change[L] #current p_change

    
    while(len(synthetic_series)<traces_len):
        #Iterate through steps

        #Select scale
        change = np.random.choice(range(n_scales+1), p = scale_change)
       
    
        if change==n_scales:
            new_cell = L


        else:
            #Move
            attractiveness = nested_dictionary[change][L[:change]]
            new_cell = L[:change+1]

            #Select new_cell
            possible_cells = [i for i in attractiveness.items() if i[0]!=new_cell]
            if len(possible_cells)==0:
                continue;
                
            k, v = list(zip(*list(possible_cells)))
            new_cell = k[np.random.choice(range(len(v)), p = np.array(list(v))/sum(list(v)))]
                
            scale=change+1
            while scale<n_scales:
                attractiveness = nested_dictionary[scale][new_cell]
                k, v = list(zip(*list(attractiveness.items())))
                new_cell = k[np.random.choice(range(len(v)), p = np.array(list(v))/sum(list(v)))]
                scale+=1
                
        #Update values
        synthetic_series.append(new_cell[-1])
        L = new_cell
        scale_change = cell_p_change[L]
        
    #Add positions
    synthetic_series = [(i,stop_coords[i]) for i in synthetic_series]
    
    return synthetic_series





def rand_entropy(sequence):
    
    '''
    Input
    -----
    Sequence (list): list of labels
    
    Output
    ------
    (float): Random entropy
    
    '''
    N = len(np.unique(sequence))
    return math.log2(N).real

def unc_entropy(sequence):
    '''
    Input
    -----
    Sequence (list): list of labels
    
    Output
    ------
    (float): Uncorrelated entropy
    
    '''
    r = np.array([*Counter(sequence).values()])
    r = r/float(sum(r))

    return - sum([i*math.log2(i) for i in r]).real


def contains(small, big):
    '''
    Input
    -----
    small (numpy array): list of labels
    big (numpy array): list of labels
    
    Output
    ------
    (bool): If big contains small return True else return False
    
    '''
    try:
        big.tostring().index(small.tostring())//big.itemsize
        return True
    except ValueError:
        return False

    
def est_entropy(l, natural_log = False):
    '''
    Input
    -----
    l (list): list of labels
    
    Output
    ------
    (float): Estimated entropy
    
    '''
    l = np.array(l)
    n = len(l)
    
    if n<=2:
        return np.nan

    sum_gamma = 0

    for i in range(1, n):
        sequence = l[:i]

        for j in range(i+1, n+1):
            s = l[i:j]
            if contains(s, sequence) != True:
                sum_gamma += len(s)
                break;

    if natural_log:
        ae = 1 / (sum_gamma / n) * math.log(n)
    else:
        ae = 1 / (sum_gamma / n) * math.log2(n)
    return ae


def upper_bound_of_predictability(S, sequence):
    '''
    Input
    -----
    S        (float): entropy
    sequence (list): list of labels
    
    Output
    ------
    (float): Upper bound of predictability (Fano inequality)
    
    '''
    N = len(np.unique(sequence))
    func = lambda x: (-(x * math.log2(x).real + (1 - x) * math.log2(1 - x).real) + (1 - x) * math.log2(N - 1).real) - S
    ub = fsolve(func, 0.99)[0]
    return ub
 

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



#Extract random number from pl with exponent beta (inverse sampling)
def extract_power_law(beta,xmin):
    u = np.random.rand()
    dt2 = xmin*(u)**(-1./(beta-1))
    return dt2


#Extract random number from truncated pl with exponent beta (inverse sampling)
def extract_truncated_pl(x0, xmin, Lambda, alpha):
    r = np.random.rand()
    while 1:
        x = xmin - (1/Lambda) * np.log(r)
        p = ((x+x0)/xmin)**-alpha
        if np.random.rand()<p:
            return x
        r = np.random.rand()

#Extract uniform number 
def random_number_jit():
    u = np.random.rand()
    return u