"""Generally useful little utility functions."""

import numpy as np
from scipy.spatial import ConvexHull
from scipy import stats
from collections import defaultdict, Counter
import matplotlib.pylab as plt
import folium
from multiprocessing import Pool
from collections import deque
import copy as cp
#from joblib import Parallel, delayed
from utils import scale_fitter_no_grid
from utils import utils
from importlib import reload
import scipy



def p_change_fit(a, alpha):
    #return 2.718281828459045**(-a * alpha)
    return (1 - a) / alpha


def _worker_bootstrap_pval(inputs):
    seed, N, source_target, cell_attractiveness, fitted_cell_p_change, source_target_prev, cell_attractiveness_prev, fitted_cell_p_change_prev = inputs
    np.random.seed(seed)
    random_indices = np.random.randint(0, N, 300)
    L = scale_fitter_no_grid.total_likelihood(source_target[random_indices], cell_attractiveness, fitted_cell_p_change)
    np.random.seed(seed+1)
    random_indices = np.random.randint(0, N, 300)
    L_prev = scale_fitter_no_grid.total_likelihood(source_target_prev[random_indices], cell_attractiveness_prev, fitted_cell_p_change_prev)
    return L, L_prev
        
def bootstrap_pval(series_prev, series, stat_test, num_iter=1000, nprocs = 10):
    """Compute bootstrap pvalue for a series.
    
    Input
    -----
        series : list of lists
        scale_index : int
            The index in the series which needs to be tested
        num_iter : int
            Number of bootstrap iterations
    """
    # Number of trips
    N = len(series) - 1
    n_scales = len(series[0])
    n_scales_prev = len(series_prev[0])
    
    if n_scales != n_scales_prev + 1:
        raise
    
    # Reshape to trips
    source_target = np.stack([series[:-1], series[1:]], axis=1)
    source_target_prev = np.stack([series_prev[:-1], series_prev[1:]], axis=1)
 
    # Cell attractiveness and cell p change of current
    cell_attractiveness = {}
    scale_fitter_no_grid.compute_cell_attractiveness(series, cell_attractiveness, 0)
    fitted_cell_p_change,_ = scale_fitter_no_grid.find_cell_p_change(source_target, n_scales, series)
    
    # Cell attractiveness and cell p change of prev
    cell_attractiveness_prev = {}
    scale_fitter_no_grid.compute_cell_attractiveness(series_prev, cell_attractiveness_prev, 0)
    fitted_cell_p_change_prev,_ = scale_fitter_no_grid.find_cell_p_change(source_target_prev, n_scales_prev, series_prev)
    # Maintain bool array of iteration test succeses and failures

    inputs = []
    for seed in range(num_iter):
        inputs.append((seed, N, source_target, cell_attractiveness, fitted_cell_p_change, source_target_prev, cell_attractiveness_prev, fitted_cell_p_change_prev))
                   
    if nprocs > 1:
        p = Pool(nprocs)
        result = p.map(_worker_bootstrap_pval, inputs)
        #result = Parallel(n_jobs=nprocs, max_nbytes=1e6)(delayed(_worker_bootstrap_pval)(inp) for inp in inputs)
    else:
        result = map(_worker_bootstrap_pval, inputs)
    
    L_vec, L_prev_vec = [], []
    for L, L_prev in result:
        L_vec.append(L)
        L_prev_vec.append(L_prev)
    
    if nprocs > 1:
        p.close()
    
    if stat_test is None:
        pval = np.mean(np.array(L_vec) >= np.array(L_prev_vec))
    else:
        pval = stat_test(L_vec, L_prev_vec)[1]
    
    return pval, L_vec, L_prev_vec
        

def haversine(points_a, points_b, radians=False):
    """ 
    Calculate the great-circle distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.

    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Using this because it is vectorized (stupid fast).
    """
    def _split_columns(array):
        if array.ndim == 1:
            return array[0], array[1] # just a single row
        else:
            return array[:,0], array[:,1]

    if radians:
        lat1, lon1 = _split_columns(points_a)
        lat2, lon2 = _split_columns(points_b)

    else:
    # convert all latitudes/longitudes from decimal degrees to radians
        lat1, lon1 = _split_columns(np.radians(points_a))
        lat2, lon2 = _split_columns(np.radians(points_b))

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2
    h = 2 * 6371e3 * np.arcsin(np.sqrt(d))
    return h  # in meters

def haversine_pdist(points, radians=False):
    """ 
    Calculate the great-circle distance bewteen each pair in a set of points.
    
    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Input
    -----
        points : array-like (shape=(N, 2))
            (lat, lon) in degree or radians (default is degree)

    Output
    ------
        result : array-like (shape=(N*(N-1)//2, ))
    """ 
    c = points.shape[0]
    result = np.zeros((c*(c-1)//2,), dtype=np.float64)
    vec_idx = 0
    if not radians:
        points = np.radians(points)
    for idx in range(0, c-1):
        ref = points[idx]
        temp = haversine(points[idx+1:c,:], ref, radians=True)
        result[vec_idx:vec_idx+temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return result

def euclidean(points_a, points_b, **kwargs):
    """ 
    Calculate the euclidian distance bewteen points_a and points_b
    points_a and points_b can be a single points or lists of points.
    """
    if len(points_a.shape) > len(points_b.shape):
        points_b = points_b.reshape([-1] + list(points_a.shape[1:]))
    elif len(points_b.shape) > len(points_a.shape):
        points_a = points_a.reshape([-1] + list(points_b.shape[1:]))
    return np.sqrt(np.sum((points_a.T - points_b.T)**2, axis=0))
    

def general_pdist(points, distance_function = haversine):
    """ 
    Calculate the distance bewteen each pair in a set of points given a distance function.
    
    Author: Piotr Sapiezynski
    Source: https://github.com/sapiezynski/haversinevec

    Input
    -----
        points : array-like (shape=(N, 2))
            (lat, lon) in degree or radians (default is degree)

    Output
    ------
        result : array-like (shape=(N*(N-1)//2, ))
    """ 
    c = points.shape[0]
    result = np.zeros((c*(c-1)//2,), dtype=np.float64)
    vec_idx = 0
        
    for idx in range(0, c-1):
        ref = points[idx]
        temp = distance_function(points[idx+1:c,:], ref, radians = False)
        #to be taken care of
        result[vec_idx:vec_idx+temp.shape[0]] = temp
        vec_idx += temp.shape[0]
    return result

def reshape_dist_arr_to_dist_matr(dist_arr, upper_only=True):
    """Take an N*(N-1)/2 long array and reshape it to a (N,N) array.
    """
    N = int((1 + np.sqrt(1 + 4 * 2 * len(dist_arr))) / 2)
    D = np.zeros((N, N)) * np.nan
    D[np.triu_indices(N, 1)] = dist_arr
    if not upper_only:
        D[np.tril_indices(N, -1)] = D.T[np.tril_indices(N, -1)]
    return D

def convex_hull(points, to_return='points'):
    """Return the convex hull of a collection of points."""
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        raise Exception("Number of unique points must be larger than 2.")
    else:
        hull = ConvexHull(points)
    if to_return == "points":
        return points[hull.vertices, :]
    if to_return == "area":
        return hull.area
    if to_return == "volume":
        return hull.volume
    if to_return == "geoarea":
        from area import area
        return area({'type':'Polygon','coordinates':[points[hull.vertices].tolist()]})

def get_scale_labels(series):
    """For each step in a multilevel walk, get the scale.

    Input
    -----
        series : list of lists

    Output
    ------
        out : list (`len(out) == len(series)`)

    Example
    -------
        >>> scale_labels = get_scale_labels(series)
        >>> scale_labels
        [1, 1, 1, 1, 1, 3, 2, 1, 1, 1, ...
    """
    n_scales = len(series[0])
    series = np.array(series)
    return [
        n_scales - np.min(np.where(list(c_i)+[True])[0])
        for c_i in series[:-1] != series[1:]
    ]

def get_container_labels(series):
    """For a multilevel walk, figure out which stop locations are in which containers at each scale

    Input
    -----
        series : list of lists

    Output
    ------
        container_labels : dict of dicts of lists
            The first level is scale, so `container_labels.keys()` will be something like `[2, 3, 4, 5]`.
            The second level is the containers in a given scale, yielding a list of all contained stop labels.

    Example
    -------
        >>> contaier_hierarchy = get_container_hierarchy(series)
        >>> contaier_hierarchy
        {   # scale
            2: {
                # container: labels
                0: [0, 1, 2, 5, 23, ...],
                ...
            }, 
            3: ...
        }
    """
    scale = len(series[0])
    containers = np.unique(series, axis=0)
    container_hiararchy = defaultdict(set)
    for cell in containers:
        for s in range(scale-1):
            container_hiararchy[tuple(cell[:s+1])].add(cell[-1])
    return container_hiararchy

def get_accuracy(series, recovered_series):
    """Compare a simulated and recovered series in terms of overall label accuracy.
    
    Input
    -----
        series : list of lists
        recovered_series : list of lists
        
    Output
    ------
        out : dict
        
    Example
    -------
        >>> performance = get_accuracy(series, recovered_series)
        >>> performance
        {'acc': 0.9708, 'baseline': 0.8539}
    """
    
    # Compute scale for each step in both series
    scales_series = get_scale_labels(series)
    scales_recovered_series = get_scale_labels(recovered_series)
    
    # Compute accuracy of a model that always guesses majority scale
    majority_scale = Counter(np.array(scales_series)).most_common(1)[0][0]
    baseline = np.mean(np.array(scales_series) == majority_scale)
    
    # Compute accuracy
    acc = np.mean(np.array(scales_series) == np.array(scales_recovered_series))
    
    return dict(acc=acc, baseline=baseline)

def get_scale_mutual_info(series, recovered_series):
    """Compare a simulated and recovered series in terms of scale by scale mutual info.

    This is how we measure whether labels are ending up in the right containers.

    Input
    -----
        series : list of lists
        recovered_series : list of lists
        
    Output
    ------
        out : dict
        
    Example
    -------
        >>> performance = get_scale_mutual_info(series, recovered_series)
        >>> performance
        {1: 0.9, 1: 0.8, 2: 0.84, ...}
    """
    pass


def colormixer(colors, weights=None):
    """Take array of colors in hex format and return the average color.
    
    Input
    -----
        colors : array of hex values
    
    Example
    -------
        >>> colormixer(['#3E1F51', '#FEE824', '#1F908B'])
        '#4af134'
    """
    def _to_hex(v):
        v_hex = hex(v)[2:]
        if len(v_hex) == 1:
            v_hex = "0" + v_hex
        return v_hex

    # Compute mean intensities for red, green and blue
    if weights is None:
        r = int(np.mean([int(c[1:3], 16) for c in colors]))
        g = int(np.mean([int(c[3:5], 16) for c in colors]))
        b = int(np.mean([int(c[5:7], 16) for c in colors]))
    else:
        r = int(sum([int(c[1:3], 16) * w for c, w in zip(colors, weights)]) / sum(weights))
        g = int(sum([int(c[3:5], 16) * w for c, w in zip(colors, weights)]) / sum(weights))
        b = int(sum([int(c[5:7], 16) * w for c, w in zip(colors, weights)]) / sum(weights))
    
    # Take mean of each and convert back to hex
    return '#' + _to_hex(r) + _to_hex(g) + _to_hex(b)

def plot_scales_histogram(series, stop_locations, colors=None, log_dist=True, density=True, distance_func=haversine):
    """Plot histogram of trip distances for all scales.
    
    Bars are colored by the average scales inside.
    
    Input
    -----
        series : list of lists
            Important! Lowest cell index must map to row in `stop_locations`!
        stop_locations : np.array (`stop_locations.shape[1] == 2`)
        colors : list of hex color strings (optional colorscheme)
       
    
    """
    nbins = 40

    # Get scale and distances of trips
    trip_scale_labels = np.array(get_scale_labels(series))
    distances = np.array([
        distance_func(stop_locations[step0[-1]], stop_locations[step1[-1]])
        for step0, step1 in zip(series[:-1], series[1:])
    ])
    
    # Remove trips with distance zero (they should not exist, but hey)
    trip_scale_labels = trip_scale_labels[distances>0]
    distances = distances[distances>0]
    
    if log_dist:
        distances = np.log10(distances)

    # Decide on some bin edges
    if not log_dist:
        hist_bins = np.logspace(np.log10(min(distances)), np.log10(max(distances)), nbins)
    else:
        hist_bins = np.linspace(min(distances), max(distances), nbins)

    # Get the bin id of each trip
    bin_ids = np.digitize(distances, bins=hist_bins)

    # Create a map between bin_id and the scale of trips inside the bin
    bin_scales = defaultdict(list)
    for bin_id, scale in zip(bin_ids, trip_scale_labels):
        bin_scales[bin_id].append(scale)

    # Order of colors that we like
    if colors is None or len(colors) < len(series[0]):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = colors+['#000000',"#ff0000",'#800080']
    
    # Compute the bar colors
    barcolors = [
        colormixer([colors[s-1] for s in scales])
        for bin_id, scales in sorted(bin_scales.items(), key=lambda kv: kv[0])
    ]

    # Plot the damn histogram!
    densities, _, patches = plt.hist(distances, bins=hist_bins, density=density)
    if not log_dist: plt.xscale("log")
    plt.yscale("log")

    # Recolor each histogram bar
    i = 0
    for density, patch in zip(densities, patches):
        if density == 0: continue
        patch.set_facecolor(barcolors[i])
        i += 1
    
    
def plot_solution_on_world_map(series, stop_locations, distance_func=haversine, filename = None):
    """Plot folium map with nested container.
    
    Input
    -----
        series : list of lists
            Important! Lowest cell index must map to row in `stop_locations`!
        stop_locations : np.array (`stop_locations.shape[1] == 2`)
        distance_func : function
        
    Example
    -------
        >>> plot_solution_on_world_map(best_branch.series, stop_locations, distance_func=euclidean)
    """
    
    if len(series[0])==1:
        print('Only one scale found: could not print map')
        return None
    series = series[:-1][(series[:-1]!=series[1:]).any(axis = 1)]

    # Get the hierarchy of cells that we want to plot
    container_labels = get_container_labels(series)
    n_scales = len(series[0])

    # HACK for non-spherical-coord points to plot on geo map anyway
    if distance_func != haversine:
        stop_locations = stop_locations / stop_locations.max()
        stop_locations[:, 1] += 90

    # Convert each cell to a convex hull
    cell_hulls = defaultdict(dict)
    for cell, stop_labels in container_labels.items():
        points = [stop_locations[label] for label in stop_labels]
        while True:
            try:
                cell_hulls[n_scales-len(cell)+1][cell] = convex_hull(np.array(points))
                break
            except: # QhullError
                points.append(points[-1] + (np.random.random(size=2)-0.5) * 1e-4)
                
    # Styling function used by folium
    def style_function(feature):
        return {"fillColor": feature["properties"]["color"], "color": feature["properties"]["color"], "weight": 1, "fillOpacity": 0.1}

    # Build GeoJSON valid dictionary
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors+['#000000',"#ff0000",'#800080']

    polygons = {"type": "FeatureCollection", "features": []}
    for s in sorted(cell_hulls.keys(), reverse=True):
        polygons["features"].append({
            "type": "Feature",
            "properties": {"color": colors[s-2]},
            "geometry": {
                "type": "Polygon",
                "coordinates": []
            }
        })
        for cell in cell_hulls[s].values():
            polygons["features"][-1]["geometry"]["coordinates"].append(
                cell[:, ::-1].tolist()
            )
    
    # Initiate folium map
    m = folium.Map(np.median(stop_locations, 0).tolist(), zoom_start=8, tiles="cartodbpositron")

    # Add data to it
    folium.GeoJson(
        polygons,
        style_function=style_function
    ).add_to(m)

    #trip_scale_labels = get_scale_labels(series)
    #for l1,l2,t in zip(series[:-1],series[1:],trip_scale_labels):
    #    c1 = stop_locations[l1[-1]]
    #    c2 = stop_locations[l2[-1]]
    #    folium.PolyLine([c1,c2], color = colors[t-1], weight = 0.5).add_to( m )
        
    #for coord in stop_locations:
    #    folium.CircleMarker(location=[ coord[0], coord[1] ], radius = 1, color = 'black').add_to( m )
    

    
    # Return it!
            
    if filename!=None:
        m.save(filename)
        
    else:
        return m

def plot_cost_function_deltas(criterion, final_sizes):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for scale, data in criterion.items():
        if scale in final_sizes:
            x, y = zip(*data)
            y = np.array(y)
            plt.plot(x, y-min(y)+1, color=colors[scale-2], label=scale)
            plt.xscale("log")
            plt.yscale("log")
            plt.axvline(final_sizes[scale], color=colors[scale-2])
    plt.legend()
    plt.ylabel("$\Delta(-\log{L})$", fontsize=12)
    plt.xlabel("Distance (meters)", fontsize=12)

def remove_consequetive_duplicates(your_list):
    """Return list of consequetively unique items in your_list.

    Input
    -----
        your_list : list

    Output
    ------
        out : list (np.array if input type is np.array)
    """
    out = [v for i, v in enumerate(your_list) if i == 0 or v != your_list[i-1]]
    if type(your_list) == np.ndarray:
        return np.array(out)
    return out

def clearnans(your_list):
    """Return list with nans removed.
    
    Input
    -----
        your_list : list

    Output
    ------
        out : list (np.array if input type is np.array)
    """
    if type(your_list) == np.ndarray:
        return your_list[~np.isnan(your_list)]
    elif type(your_list) is list:
        return [v for v in your_list if not np.isnan(v)]
    
def last_index(arr, item):
    """Return not first (as `.index`) but last index of `item` in `arr`.
    """
    return len(arr) - arr[::-1].index(item) - 1

def nth_index(arr, item, n):
    """In a list that contains an item multiple times, return the index
    of its n'th mention.
    """
    return np.where(np.array(arr) == item)[0][n]

def split_to_ranges(arr):
    """Split a list into a list of lists so each list is a range.

    Example
    -------
        >>> split_to_ranges([0, 1, 2, 0, 1, 0, 1, 2])
        [[0, 1, 2], [0, 1], [0, 1, 2]]
    """
    ranges = [[]]
    c = 0
    for v0, v1 in zip(arr[:-1], arr[1:]):
        ranges[c].append(v0)
        if v0 + 1 != v1:
            ranges.append([])
            c += 1
    ranges[c].append(v1)
    return ranges

def exists(var):
    try:
        var
        return True
    except NameError:
        return False

def k_min(arr, k, filter_nans=True):
    """Return the indices of the k smallest values in a 2d array.

    Input
    -----
        arr : np.array (2d)
        k : int (> 0)

    Output
    ------
        out : list of tuples

    Example
    -------
        >>> arr = np.arange(0, 9).reshape(3, 3)
        >>> k_min(arr, 2)
        [(0, 0), (0, 1)]
    """
    n_cols = arr.shape[1]
    k = min(k, arr.shape[0] * arr.shape[1]-1)
    flat_arr = arr.reshape(-1)
    indices = [
        (ind // n_cols, ind % n_cols)
        for ind in np.argpartition(flat_arr, k)[:k]
    ]
    if filter_nans:
        return [
            (i, j) for i, j in indices
            if not np.isnan(arr[i, j])
        ]
    return indices

def default_to_regular(d):
    """Recursively convert nested defaultdicts to nested dicts.

    Source: http://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d




def flatten(container):
    """Flatten arbitrarily nested list of lists.
    """
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
            
def furthest(location, list_of_locations, map_latitude, map_longitude):
    '''
    Find furthest location from "location" within list_of_locations.
    
    location: location id
    list_of_locations: list of locations id
    map_latitude: latitude dictionary
    map_longitude dictionary of longitudes
    
    '''
    loc_ = (map_latitude[location],map_longitude[location])
    list_ = [(map_latitude[i],map_longitude[i]) for i in list_of_locations]
    index = np.argmax(haversine(loc_, list_))
    return list_of_locations[index]

find_depth = lambda L: isinstance(L, list) and max(map(find_depth, L))+1

def find_keystone_trips(sequence, map_latitude, map_longitude):
    '''
    sequence : list of locations
    map_latitude: dictionary of latitudes
    map_longitude: dictrionary of longitudes
    all_trips: if True returns all the trips (ex: ['a','b','c','b','a']--> [['a','c'],['b','c'],['c','b'],['c','a']])
               (it select the furthest location for each trip. In this case 'c' is the furthest from 'a')
               
               
    
    
    Return: list of tuples (origin, destination)
            returns trips to keystones  (ex: ['a','b','c','b','a']--> [['a','b'],['b','c'],['c','b'],['b','a']])
            it select the furthest location in case of a trip that contains no keystone (c in the example above)
    '''
    
    
    trips = []
    output = deque()
    q = deque() #queue of opened trips
    for location in sequence:
        q2 = cp.copy(q) #iterate trough opened trips
        q.append(location)
        output.append(location)
        trip = deque()
        n=0
        while q2:
            n+=1
            old_location = q2.pop()
            #If I find the location previously I close the trip
            if old_location==location:
                for i in range(n):
                    q.pop()
                    
                arrival = output.pop()
                loc_trip = None
                while (loc_trip!=location) :
                    loc_trip = output.pop()
                    trip.appendleft(loc_trip)
                    
                departure = trip.popleft()
                output.append(departure)
                output.append(list(trip))
                output.append(arrival)
                
                    
                destination=None
                for item in list(trip):
                    if type(item)==list:
                        break;
                    destination = item
                if destination==None:
                    destination = furthest(departure, list(flatten(list(trip))), map_latitude, map_longitude)
                n = find_depth(list(trip))
                if n!=1:
                    trips.insert(-n,(departure, destination))
                else:
                    trips.append((departure, destination))
                trips = trips+[(destination, departure)]
                break;
    return np.array([i for (i,j) in trips])


            
def cophenetic_distance(point_a, point_b, radians = False):
    '''
    Compute the cophenetic distance between point_a and point_b.
    Input
    -------
        point_a: list
                Hierarchical description of a point.
        point_b: list
                Hierarchical description of a point.
    
    Output
    -------
        int: cophenetic distance between the two points
    
    '''
    s = len(point_a[0])
    arr = point_a!=point_b
    mask = arr!=0
    return s - np.where(mask.any(axis=1), mask.argmax(axis=1), s)


def cophenetic_correlation(original_series, recovered_series):
    '''
    Compute the cophenetic correlations between two list of points.
    Input
    -------
        original_series: list of lists
                Sequence of points in hierarchical description.
                
        recovered_series: list of lists
                Sequence of the same points with a different hierarchical description.
    
    Output
    -------
        int, float: cophenetic correlation between the two hierarchical descriptions, p-value
    

    '''
    s1 = len(original_series[0])
    s2 = len(recovered_series[0])

    unique_zipped_series = np.unique(np.concatenate([original_series,recovered_series],axis =1),axis = 0)

    distance_1 = utils.general_pdist(unique_zipped_series[:,:s1], cophenetic_distance)
    distance_2 = utils.general_pdist(unique_zipped_series[:,s1:], cophenetic_distance)

    return scipy.stats.pearsonr(distance_1,distance_2)