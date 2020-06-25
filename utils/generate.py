import numpy as np
from utils import scale_fitter_no_grid
import itertools


def recover_parameters_from_fitted_trace(trace):
    '''Given a trace, recover parameters a and p.
    
    Input
    -----
    trace: list of lists (e.g.[ [1,2,3] , [1,2,1], ...., [2,1,1]])
           Sequence of locations in hierarchical form.
    
    
    Output
    ------
    nested_dictionary: (dict)
        Gives the attractiveness of each container.
    
    cell_p_change: (dict)
        Gives the probability of changing at any level-distance for each cell.
        
    '''
    
  
    #Create the source_target_list and compute the parameters of the model
    source_target = np.stack([trace[:-1], trace[1:]], axis=1)
    (proba_dist, proba_dist_counts), cell_attractiveness, cell_p_change, _ = scale_fitter_no_grid.compute_likelihood(source_target, return_all_values=True)

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
    return nested_dictionary, cell_p_change


def generate_trace(nested_dictionary, cell_p_change, size, initial_position = None):
    '''
    Generate a synthetic trace starting from a sequence of locations with the corresponding scale structure
    
    Input
    -----
    nested_dictionary: (dict)
        Gives the attractiveness of each container.
    cell_p_change: (dict)
        Gives the probability of changing at any level-distance for each cell.
    size: (int)
        Length of the sythethic sequence.
    initial position: (list)
        Initial position
 

    Output
    ------
    synthetic_trace: list of lists (e.g.[ [1,2,3] , [1,2,1], ...., [2,1,1]])
    
    '''
    
    #Recover parameters
    traces_len = int(size)
    n_scales = len(list(cell_p_change.values())[0]) - 1


    #Initialize synthetic trace
    if initial_position==None:
        locs = range(len(cell_p_change.keys()))
        l = np.random.choice(locs)
        initial_position = list(cell_p_change.keys())[l]
        
    L = tuple(initial_position) #current cell
    synthetic_series = [L[-1]] #sequence of cells
    scale_change = cell_p_change[L] #current p_change

    
    while(len(synthetic_series)<traces_len):
        #Iterate through steps

        #Select level
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

    return synthetic_series
