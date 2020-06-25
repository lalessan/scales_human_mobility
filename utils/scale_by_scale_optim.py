import numpy as np
import pandas as pd
from collections import defaultdict
from utils import scale_fitter_no_grid
from copy import deepcopy as dc
from utils import utils
from bisect import bisect
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

class ScalesOptim:
    """Scales optimizer class.
    
    Input
    -----
        labels : list
            Sequence of stops. Example [0, 1, 1, 0, 10, 2, ...]
        stop_locations : np.array (shape=(len(set(labels))), 2))
            (lat, lon) Coordinates of locations. Row index must correspond to label index.
        distance_func : callable 
            Function to compute distances
        min_dist : float (> 1) 
            Consider linkage solutions such that distance[i]>min_distance*distance[i-1]
        min_diff : int
       
        siglvl : float (0 < siglvl
            Significance value for tests.
        stat_test: callable
            Function that takes as input two lists of values. Statistical test.
        bootstrp: bool
            Run the bootstrap routine or not.             
        information_criterion : str (None, 'AIC' or 'BIC')
            If different than None, choose between AIC or BIC.
        n_procs: int
            Number of processes
        linkage_method: str
        verbose: bool
        bootstrap_iter: (int)
            Number of bootstrap iterations.
            

    """
    def __init__(self, 
                 labels, 
                 stop_locations,  
                 distance_func=utils.haversine, 
                 min_dist=1.2, 
                 min_diff=0, 
                 siglvl=0.05, 
                 stat_test=stats.ks_2samp, 
                 bootstrap=False, 
                 repeated_trips=True, 
                 information_criterion="AIC", 
                 nprocs=1, 
                 linkage_method="complete", 
                 verbose=True, 
                 bootstrap_iter=200):

        self.stop_locations = stop_locations
        self.distance_func = distance_func
        self.labels = labels
        self.min_dist = min_dist
        self.min_diff = min_diff
        self.siglvl = siglvl
        self.stat_test = stat_test
        self.bootstrap = bootstrap
        self.repeated_trips = repeated_trips
        self.information_criterion = information_criterion
        self.nprocs = nprocs
        self.verbose = verbose
        self.num_iter=bootstrap_iter
        self.alphas = []
        
        #Find possible clusterings
        pdist = utils.general_pdist(self.stop_locations, distance_function=distance_func)
        self.Z = linkage(pdist, method=linkage_method)
        
        # Merge distances for all branches to follow
        self.merge_d = []
        for i, d in enumerate(self.Z[:-1, 2]):
            if i == 0 or d > self.min_dist * self.merge_d[-1]:
                self.merge_d.append(d)
    
        self.max_d = self.Z[-1,2]
        
        
        
    def _worker(self, inputs):

        n, max_Rs, label_to_cell_map, final_series, final_scales, n_params_prev = inputs

        # Update series
        candidate_series = []
        scale_index = bisect(sorted(final_scales.values()), n)
        for element in final_series:
            a = dc(element)
            a.insert(scale_index, label_to_cell_map[element[-1]].item())  # `.item()` yields native python int
            candidate_series.append(a)

        # Compute likelihood
        source_target = np.stack([candidate_series[:-1], candidate_series[1:]], axis=1)
        (proba_dist, proba_dist_counts), _, _, alphas = scale_fitter_no_grid.compute_likelihood(source_target, return_all_values=True)
        LL = sum(-np.log(proba_dist))
        
        if not self.repeated_trips:
            proba_dist_proba_dist_counts = set(zip(
                proba_dist,
                map(str, proba_dist_counts)
            ))
            proba_dist = np.array([v0 for v0, v1 in proba_dist_proba_dist_counts])
            
        #Compute the criterion
        n_params = n_params_prev + len(set([i[scale_index] for i in candidate_series])) + 1

        if self.information_criterion == 'AIC':
            criterion = 2 * n_params + 2 * LL 
        elif self.information_criterion == 'BIC':
            criterion = np.log(len(self.labels)-1) * n_params + 2 * LL
        else:
            criterion = LL

        return candidate_series, n_params, LL, proba_dist, criterion, n, max_Rs, label_to_cell_map, alphas


    def find_best_scale(self):
        """
        Run the optimization routine and find the best combinations of the scales.
        
        """        
        #Find L_min
        series = [[c] for c in self.labels]
        source_target = np.stack([series[:-1], series[1:]], axis=1)
        (proba_dist_min, _), _, _, alphas_min = scale_fitter_no_grid.compute_likelihood(source_target, return_all_values=True)
        L_min = sum(-np.log(proba_dist_min))

        #Initialize all values
        final_series = dc(series)
        final_proba_dist = proba_dist_min
        
        scales = dict() 
        sizes = dict()
        
        final_scales = dc(scales)
        final_sizes = dc(sizes)
        final_series = dc(series)
        final_alphas = dc(alphas_min)
        
        improvement = True
        scale = 2
        likelihoods = defaultdict(list)
        criterion_s = defaultdict(list)
        
        n_params_min = n_params_prev = len(set(self.labels)) + 1
        proba_dist_prev = dc(proba_dist_min)

        if self.information_criterion == 'AIC':
            criterion_min = 2 * n_params_min + 2 * L_min
            
        elif self.information_criterion == 'BIC':
            criterion_min = np.log(len(self.labels)-1) * n_params_min + 2 * L_min
        
        else:
            criterion_min = L_min 

        #Add a scale unntil there is no more improvement
        if self.verbose: print('Searching for minimum at scale {}:\n'.format(scale))
        while improvement:
            improvement = False

            #Try all possible clusterings
            inputs = []
            for n, max_Rs in enumerate(reversed(self.merge_d)):
                label_to_cell_map = dict(enumerate(fcluster(self.Z, max_Rs, criterion='distance')))
                if len(set(scales.values()) & set(range(n-self.min_diff, n+1+self.min_diff))) == 0:  # This obscure line just checks if n or indices adjacent to n (tunable parameter `min_diff`) are already chosen.
                    inputs.append((n, max_Rs, label_to_cell_map, final_series, final_scales, n_params_prev))
            
            if self.nprocs > 1:
          
                result = Parallel(n_jobs=self.nprocs, max_nbytes=1e6)(delayed(self._worker)(inp) for inp in inputs)
            else:
                result = map(self._worker, inputs)
                
            for candidate_series, n_params, LL, proba_dist, criterion, n, max_Rs, label_to_cell_map, alphas in result:
                d_criterion = criterion - criterion_min
            
                # Save likelihood and criterion
                likelihoods[scale].append((max_Rs, LL))
                criterion_s[scale].append((max_Rs, criterion))

                # Print user output
                if self.verbose:
                    print("    It: %d/%d" % (n, len(self.merge_d)-1), end=" ")
                    if max_Rs < 1000: print("| d: %.01f m" % max_Rs, end=" ")
                    else: print("| d: %.01f km" % (max_Rs/1000), end=" ")
                    print("| L: %.01f" % LL, end=" ")
                    print('| p: '+','.join([': ['.join([str(i),','.join(["%.02f" % k for k in v]+["] "])]) for i,v in alphas.items()]), end = " ")
                    if self.information_criterion:
                        print("| %s: %.01f" % (self.information_criterion, criterion), end=" ")
                        print("| ∆%s: %.01f" % (self.information_criterion, d_criterion))
                    else:
                        print("| ∆L: %.01f" % d_criterion)
                
                # Check if Likelihood has improved and criterion is positive
                if d_criterion < 0:
                    improvement = True
                    L_min = LL
                    criterion_min = criterion
                    proba_dist_min = proba_dist
                    scales[scale] = n
                    sizes[scale] = max_Rs
                    series = candidate_series
                    n_params_min = n_params
                    alphas_min = alphas

            if improvement:
                if self.verbose:
                    print('\nFound minimum at   d:', end=" ")
                    if sizes[scale] < 1000: print("%.01f m" % sizes[scale])
                    else: print("%.01f km" % (sizes[scale]/1000))
                    print('                   L: %.01f' % L_min)
                    if self.information_criterion:
                        print('                 %s: %.01f' % (self.information_criterion, criterion_min), end="\n\n")
                    else:
                        print('', end="\n\n")
                    
                # Compute p-value
                if self.siglvl is not None:
                    if self.verbose:
                        print("Result of statistical test:")
                    if self.bootstrap:
                        pval, L_vec, L_prev_vec = utils.bootstrap_pval(final_series, series, self.stat_test, num_iter = self.num_iter, nprocs = self.nprocs)
                    else:
                        pval = self.stat_test(-np.log(final_proba_dist), -np.log(proba_dist_min))[1]
                    if pval > self.siglvl:
                        del likelihoods[scale]
                        del criterion_s[scale]
                        improvement = False
                
                if self.verbose and self.bootstrap:
                    plt.figure(figsize=(6, 2))
                    plt.hist(L_vec, label="Candidate", alpha=0.5)
                    plt.hist(L_prev_vec, label="Previous", alpha=0.5)
                    plt.xlabel("L", fontsize=12)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=12)
                    plt.show()
                    print("    p =", pval)
                    if pval < self.siglvl:
                        print("    --> Rejecting null hypothesis.", end="\n\n")
                    else:
                        print("    --x Cannot reject null hypothesis.", end="\n\n")
            
            if improvement:
                scale+=1
                n_params_prev = n_params_min
                proba_dist_prev = proba_dist_min
                
                final_scales = dc(scales)
                final_series = dc(series)
                final_sizes = dc(sizes)
                final_proba_dist = dc(proba_dist_min)
                final_alphas = dc(alphas_min)

                if self.verbose: print('Searching for minimum at scale {}:\n'.format(scale))
            else:
                if self.verbose:
                    print('Could not improve beyond scale %d. Optimization ends.' % (scale-1))
                else:
                    print('Found %d scales' % (scale-1))
                    
                
                
        # Add code that sorts scale indices by size
        
        return final_series, final_scales, likelihoods, criterion_s, final_sizes, final_proba_dist, final_alphas
