import numpy as np
import itertools
from collections import Counter
import warnings

#Ignore "mean of empty slice error" in populate p_change
warnings.filterwarnings("ignore", category=RuntimeWarning) 




def compute_cell_attractiveness(series, cell_attractiveness, n=0):
    '''
    Recursive function to populate the `cell_attractiveness` dictionary.
    
    Input
    -----
        series: list 
            Sequence of locations
        cell_attractiveness: dict
            Dictionary to populate
        n: int 
            Level index
    '''
    series = sorted(series)
    n_scales = len(series[0])
    
    #group the elements by id (up to scale n) and count the number of occurances
    groups = itertools.groupby(series, key=lambda x: tuple(x[:n+1]))
    key_groups = [(key, len(list(group))) for key, group in groups]
    
    #update the dictionary: keys are cell ids, and values are attractiveness
    sum_values = sum([v for k, v in key_groups])
    key_groups = [(k, min(v/sum_values, 0.99)) for k, v in key_groups]

    cell_attractiveness.update(key_groups)
    
    #if we have not completed for all scales, apply this function recursively
    if n < n_scales - 1:
        for key, group in itertools.groupby(series, key=lambda x: tuple(x[:n+1])):
            compute_cell_attractiveness(list(group), cell_attractiveness, n+1)
 
            

def find_cell_p_change(source_target, n_scales, series):
    '''
    Find the probability of transitioning at given scale for all locations.
    
    Input
    -----
        source_target: numpy 2d-array (N, n_scales, n_scales)
            The original sequence of cell-ids
        n_scales: (int)
            Number of levels
        
    Output
    ------
        cell_p_change_dict: (dict)
            Dictionary with locations as keys, and prob of transitioning at any distance-level as values
            
        cell_p_change_dict_by_d: (dict)
           Dictionary with distance from home as key, prob of transitioning at any distance-level as value
        
    
    '''

    #Find where location changes
    change = source_target[:, 0] != source_target[:, 1]    
    
    #Find  most important location
    home = sorted(Counter([tuple(i) for i in series]).items(), key = lambda x:x[1], reverse = True)[0][0]
    
    #for each location find how many levels it is far from home    
    locations = np.unique(series,axis = 0)
    distance_from_home = np.argmax(locations!=np.array([home]*len(locations)), axis = 1)
    dictionary = dict(zip([tuple(i) for i in locations], distance_from_home))
    dictionary[tuple(home)] = n_scales
 

    #group all transitions by distance from home (of the source)
    source_target = sorted(source_target, key = lambda x:dictionary[tuple(x[0])])    
    groups = itertools.groupby(source_target, key=lambda x: dictionary[tuple(x[0])])
    
    
    #Create a dictionary of cell_p_change for each location.
    cell_p_change_dict = {}
    cell_p_change_dict_by_d = {}
    for key, group in groups:
        group = np.array(list(group))
        
        #At which index the transition occurs
        change = group[:, 0] != group[:, 1]
        change_indexes = np.argmax([list(i)+[True] for i in list(change)], axis=1)
        d = Counter(change_indexes)
        
        #Compute values and update dictionary
        cell_p_change = [d.get(n,0.001) for n in range(n_scales+1)]
        cell_p_change = [i/float(sum(cell_p_change)) for i in cell_p_change]
        
        cell_p_change_dict_by_d[n_scales - key] = cell_p_change
        
        for location in group[:,0]:
            cell_p_change_dict[tuple(location)] = cell_p_change
            
    #Fix eventually the last location in the series (it could be that it did not appear elsewhere)
    last_location = tuple(series[-1])
    cell_p_change_dict[last_location] = cell_p_change_dict_by_d[n_scales - dictionary[last_location]]
   
    
    return cell_p_change_dict, cell_p_change_dict_by_d

   
    

def total_likelihood(source_target, cell_attractiveness, cell_p_change, return_all_values=False):
    '''
    Compute the total likelihood given all parameters.
    
    source_target: (list of lists)
        List of transitions.
    cell_attractiveness: (dict)
        Assigning attractiveness to each container
    cell_p_change: (list)
        Dictionary containing the probability to travel at any given level-distance for each location.
    return_all_values: (bool)
        If true, returns the likelihood associated to each transition.
    
    '''
    
    n_scales = len(source_target[0][0])
    unique, counts = np.unique(source_target, return_counts=True, axis=0)

    # Create sequence of origin-destinations
    change = unique[:, 0] != unique[:, 1]
    change_indexes = np.argmax([list(i)+[True] for i in change], axis=1)

    #Compute adjusted bernullis from dict
    unique_cell_p_change = np.array([cell_p_change[tuple(cell)] for cell in unique[:, 0]])

    
    #Compute cell attractiveness
    attractiveness_source = np.array([[cell_attractiveness[tuple(a[:n+1])] for n in range(n_scales)] for a in unique[:, 0]])
    attractiveness_target = np.array([[cell_attractiveness[tuple(a[:n+1])] for n in range(n_scales)] for a in unique[:, 1]])

    normalized_attr = attractiveness_target/(1 - attractiveness_source)
    normalized_attr = np.array([list(i)+[1] for i in normalized_attr])
    
    #Probabilities of scale changes

    
    p_s = np.clip(np.choose(change_indexes, unique_cell_p_change.T), 0.01, 0.99)

    #Compute prob of selecting a cell
    cell_probabilities_1 = np.choose(change_indexes, normalized_attr.T)    

    #Prob of selecting all other cells
    cell_probabilities_2 = np.array([np.prod(k[change_indexes[n]+1:]) for n,k in enumerate(attractiveness_target)])

    # Compute total likelihood
    if not return_all_values:
        r = p_s*cell_probabilities_1*cell_probabilities_2
        r = np.concatenate([[i]*c for i, c in zip(r, counts)])
        return sum(-np.log(r))
    
    else:
        r = p_s*cell_probabilities_1*cell_probabilities_2

        return (
            np.concatenate([[i]*c for i, c in zip(r, counts)]),    # array([6.02322201, 6.02322201, ..., 8.23421385, 4.28297013])  # probabilities
            np.concatenate([[i]*c for i, c in zip(unique, counts)]) # array([[[0, 3], [390, 2]], [[390, 2], [3, 21]], ... ])        # transitions
        )

    
def compute_likelihood(source_target, return_all_values=False):
    """Given a series of transitions (described as a hierarchy) compute the likelihood of the hiearchical partitioning.
    
    Input
    -----
        source_target : list of lists
            The series of trips in hierarchical description.
            The largest scale is the first value, the smallest scale is the last value.
    Output
    ------
        L: float
            The value of the likelihood.
        cell_attractiveness: dict
            Dictionary containing the attractiveness of cell_ids at all scales. 
        cell_p_change: dict
            Dictionary containing the out-transition probabilities for each container.
        cell_p_change_by_d: list of floats
            Dictionary containing the out-transition probabilities for each distance-from-home.
    """
    
    n_scales = len(source_target[0][0])
    series = source_target[:, 0].tolist() + [source_target[-1, 1].tolist()]

    # Estimate the cell attractiveness, like {(2, ): 0.8, (20, ): 0.1, ..., (2, 15): 0.9, (2, 4): 0.05, ...}
    cell_attractiveness = {}
    compute_cell_attractiveness(series, cell_attractiveness, 0)

    # Find cell p change from data
    cell_p_change, cell_p_change_by_d = find_cell_p_change(source_target, n_scales, series)    

    # Compute likelihood
    L = total_likelihood(source_target, cell_attractiveness, cell_p_change, return_all_values)
    return L, cell_attractiveness, cell_p_change, cell_p_change_by_d


