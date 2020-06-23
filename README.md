# The scales of Human Mobility
This package implements the model described in 
[** article missing for now ** ]

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

The package allows to extract charachteristic "containers" in the mobility trace.

```Python
>>> from scales_human_mobility import ScaleModel
>>> model = ScaleModel()
>>> containers = model.fit_predict(data)
```

Solutions can be plotted using:

```Python
>>> from infostop import plot_map
>>> folmap = plot_map(model)
>>> folmap.m
```

Plotting this onto a map:
