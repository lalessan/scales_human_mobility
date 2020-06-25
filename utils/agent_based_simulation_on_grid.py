from collections import defaultdict
import numpy as np
import functools

def extract_discrete_pl(beta, xmax, xmin = 1):
    '''Return an integer number between xmin-1 and xmax extracted from a power law distribution P(x)~x^(-beta).    
    
    Input
    -----
    xmin: (float) (>=1)
        Minimum value.
    xmax: (float)
        Maximum value.
    beta: (float)
        Exponent.
        
    Output
    ------
    (float): value from the power law.
    
    '''
    
    u = np.random.rand()
    dt2 = ((u)*(xmax**(1-beta)-xmin**(1-beta))+xmin**(1-beta))**(1./(1-beta))
    
    return int(dt2) - 1


def scales_to_loc(scales_vector,comb, space = True):
    '''Return (x,y) coordinates on the grid, given the vector containing the coordinates of each scale.   
    
    Input
    -----
    scales_vector: (list of int)
        Location (hierarchical description).
    comb: (list of int)
        Size of each level (in terms of number of containers).
    space: (bool)
        If true, leaves some space between two neighbouring containers.
        
    Output
    ------
    (float, float): x,y coordinates.
    
    '''
    S = len(scales_vector)
    ticks_prod = [1]+list(np.cumprod(comb))
    size_of_the_box = ticks_prod[-1]
    
    x = int(sum([np.floor(size_of_the_box / ticks_prod[1 + s]) * ((c) % comb[s]) for (s,c) in enumerate(scales_vector)]))
    y = int(sum([np.floor(size_of_the_box / ticks_prod[1 + s]) * np.floor((c) / comb[s]) for (s,c) in enumerate(scales_vector)]))

    #Add space
    if space:
        space_x = sum([1*int(x/k) for (k) in ticks_prod])-x
        space_y = sum([1*int(y/k) for (k) in ticks_prod])-y
        x+=space_x
        y+=space_y
    
    return x,y

def random_dictionary(n_ticks):
    '''Return a dictionary where keys are the range from 0 to n_ticks and values are the shuffled values.
    
    Input
    -----
    n_ticks: (int)
        Number of containers
        
    Output
    ------
    (dict): 
        For each key it assign a shuffled label.
    
    '''
    return dict(zip(range(n_ticks**2),np.random.choice(range(0,n_ticks**2-1),n_ticks**2-1, replace=False)))


def initialize_simulation(comb, n_scales, c0, c1,d):
    '''Inititalize the agent based model simulation.
    
     Input
    -----
    comb: (list of int)
        Number of containers per level.
    n_scales: (int)
        Number of levels. 
    c0: (float)
        Parameter to compute the distribution of attractiveness.
    c1: (float)
        Parameter to compute the distribution of attractiveness.
    d: (float)
        Paramenter to compute the distribution of probabilities of travelling at given level.
        
    Output
    ------
    (dict): 
        Contains the positions of containers.
    (tuple):
         Initial position.
    (dict):
         Probabilities of transitioning at a given scale.
    (list):
        Parameter of the power law distribution of attractiveness (one parameter per level).
    
    '''
    
    #parameters of the bernullis and power-laws distributions
    p_change = [np.exp(-d*s) for s in reversed(range(1,n_scales+1))]
    p_change = p_change/sum(p_change)
    betas = [c0+c1*i for i in reversed(range(n_scales))]
    
    nested_dictionary = []
    my_initial_position = []
    for n in range(0, n_scales):
        #This is a defaultdict (if it does not exist for a given cell-id it will be created)
        n_ticks = comb[n]
        aa = defaultdict(functools.partial(random_dictionary, n_ticks))
        nested_dictionary.append(aa)
        #extract a random value for each scale
        random_number = extract_discrete_pl(betas[n], n_ticks**2)
        #find its position using the dictionary
        scale_position = nested_dictionary[n][tuple(my_initial_position)][random_number]
        #append to the initial position
        my_initial_position.append(scale_position)
    
    return nested_dictionary, my_initial_position, p_change, betas


def run_simulation(iterations, comb, n_scales, c0, c1,d):
    ''' Simulate the movements of an agent on a nested grid.
    
    Input
    -----
    iterations (int): 
        Number of displacements.
    comb (list of int):
        Number of containers in each level.
    c0: (float)
        Parameter to compute the distribution of attractiveness.
    c1: (float)
        Parameter to compute the distribution of attractiveness.
    d: (float)
        Paramenter to compute the distribution of probabilities of travelling at given level.
    
    Output
    ------
    (list of tuples): x,y coordinates of the trace
    (list of tuples): trace in the hierarchical description
    (list): probability of travelling at given level-distance
    (list): parameters of the power laws describing the distribution of attractiveness at different levels
    '''


    #initialize simulation
    nested_dictionary, my_previous_position, p_change, betas = initialize_simulation(comb, n_scales, c0, c1,d)
    my_positions = [scales_to_loc(my_previous_position,comb)]
    my_positions_scales = [my_previous_position]
    
    #run simulation
    for t in range(iterations):
        my_position = []     
        
        #CHOSE l*
        l = np.random.choice(a = range(len(p_change)), size = 1, p= p_change)[0]
        
        #Copy larger containers
        for i in range(0,l):
            my_position.append(my_previous_position[i])
        

        #select new container (!= from current)
        old_cell = my_previous_position[l]
        new_cell = old_cell
        while(new_cell==old_cell):
            random_number = extract_discrete_pl(betas[l], comb[l]**2)
            new_cell = nested_dictionary[l][tuple(my_position)][random_number]
        my_position.append(new_cell)
        
        #Select smaller containers
        for n in range(l+1, len(p_change)):
            random_number = extract_discrete_pl(betas[n], comb[n]**2)
            new_cell = nested_dictionary[n][tuple(my_position)][random_number]
            my_position.append(new_cell)

        my_previous_position = my_position   
        my_positions.append(scales_to_loc(my_position,comb))
        my_positions_scales.append(my_position)
    return my_positions, my_positions_scales, nested_dictionary, p_change, betas


class ContainerModel():
    """This is a container model"""
    
    def __init__(self, 
                 comb,
                 c0,
                 c1,
                 d
                 ):
        self.c0 = c0
        self.c1 = c1
        self.d = d
        self.comb = comb
        self.n_scales = len(comb)
        
    def run_simulation(self, iterations):
        self.positions, self.positions_scales, self.nested_dictionary, self.bernullis, self.betas = run_simulation(iterations, self.comb, self.n_scales, self.c0, self.c1, self.d)
        

