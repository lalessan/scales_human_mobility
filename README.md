# The scales of Human Mobility
This package implements the model described in 
Alessandretti, L., Aslak, U. & Lehmann, S. The scales of human mobility. Nature 587, 402–407 (2020). https://doi.org/10.1038/s41586-020-2909-1

## Usage
Given a location trace such as:

```Python
>>> data 
array([[ 55.75259295,  12.34353885, 1581401760 ],
       [ 55.7525908 ,  12.34353145, 1581402760 ],
       [ 55.7525876 ,  12.3435386 , 1581403760 ],
       ...,
       [ 63.40379175,  10.40477095, 1583401760 ],
       [ 63.4037841 ,  10.40480265, 1583402760 ],
       [ 63.403787  ,  10.4047871 , 1583403760 ]])
```

The package allows to extract charachteristic "containers" in the mobility trace, generate synthetic data, and visualize solutions on a map. (see example: 2_Fit_data_and_generate.ipynb)


![img](https://github.com/lalessan/scales_human_mobility/blob/master/containers_example.png)




Further, it allows to generate realistic synthetic data on a grid (see example: 3_Generate_data_on_a_grid.ipynb)

