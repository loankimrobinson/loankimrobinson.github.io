---
layout: post
title:  "Scaling and Preprocessing Data with Scikit-Learn for Machine Learning"
categories: [Tutorial, Python, Machine Learning]
tags: [favicon, python, ml, scikit]
---


Data scaling is a data preprocessing step for numerical features. Many
machine learning algorithms such as e Gradient descent methods, KNN
algorithm, linear and logistic regression, Principle Component Analysis
(PCA), etc require data scaling for good results.

Explore more at <https://scikit-learn.org>

1.  Standard Scaling (StandardScaler)
2.  Min-Max Scaling (MinMaxScaler)
3.  Max-Abs Scaling (MaxAbsScaler)
4.  Robust Scaling (RobustScaler)
5.  Power Transformation (Yeo-Johnson)
    PowerTransformer(method=\"yeo-johnson\")
6.  Power Transformation (Box-Cox) PowerTransformer(method=\"box-cox\")
7.  Quantile Transformation (Uniform pdf)
    QuantileTransformer(output_distribution=\"uniform\")
8.  Quantile Transformation (Gaussian pdf)
    QuantileTransformer(output_distribution=\"normal\")
9.  Sample-wise L2 Normalizing (Normalizer)
10. Binarize (Binarizer)
11. Spline Transformation (SplineTransforme)

``` python
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.datasets import fetch_california_housing
os.chdir("/Users/lrobinson/Python_Projects")
from normalize_plot_0624 import make_plot
from normalize_plot_0624 import getdata
```

# Loading California Housing

For this study, let\'s take California housing as an example. The goal
is to compare data with and without scaling and observe the effects of
the presence of outliers.

``` python
california_housing = fetch_california_housing(as_frame=True)
print("\n\n========================DATA TYPE==============================\n\n")
print(type(california_housing))
print("\n\n========================DESCRIPTION==============================\n\n")
print(dir(california_housing))
```



    ========================DATA TYPE==============================


    <class 'sklearn.utils._bunch.Bunch'>


    ========================DESCRIPTION==============================


    ['DESCR', 'data', 'feature_names', 'frame', 'target', 'target_names']

# Data Insights

## Data Description and Summary

``` python
print("\n\n========================DATA DESCRIPTION==============================\n\n")
print(california_housing.DESCR)
```

    ========================DATA DESCRIPTION==============================


    .. _california_housing_dataset:

    California Housing dataset
    --------------------------

    **Data Set Characteristics:**

        :Number of Instances: 20640

        :Number of Attributes: 8 numeric, predictive attributes and the target

        :Attribute Information:
            - MedInc        median income in block group
            - HouseAge      median house age in block group
            - AveRooms      average number of rooms per household
            - AveBedrms     average number of bedrooms per household
            - Population    block group population
            - AveOccup      average number of household members
            - Latitude      block group latitude
            - Longitude     block group longitude

        :Missing Attribute Values: None

    This dataset was obtained from the StatLib repository.
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of dollars ($100,000).

    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).

    A household is a group of people residing within a home. Since the average
    number of rooms and bedrooms in this dataset are provided per household, these
    columns may take surprisingly large values for block groups with few households
    and many empty houses, such as vacation resorts.

    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.

    .. topic:: References

        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297

``` python
data = pd.DataFrame(data= np.c_[california_housing['data'], california_housing['target']],
                     columns= california_housing['feature_names'] + california_housing['target_names'])
data.head()
```


```{=html}
<div class="p-Widget jp-RenderedHTMLCommon jp-RenderedHTML jp-mod-trusted jp-OutputArea-output" data-mime-type="text/html">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
    
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>
```

``` python
print("\n\n==================================DATA SUMMARY==============================\n\n")
data.describe()
```




    ==================================DATA SUMMARY==============================

```{=html}
<div class="p-Widget jp-RenderedHTMLCommon jp-RenderedHTML jp-mod-trusted jp-OutputArea-output" data-mime-type="text/html">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.870671</td>
      <td>28.639486</td>
      <td>5.429000</td>
      <td>1.096675</td>
      <td>1425.476744</td>
      <td>3.070655</td>
      <td>35.631861</td>
      <td>-119.569704</td>
      <td>2.068558</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.899822</td>
      <td>12.585558</td>
      <td>2.474173</td>
      <td>0.473911</td>
      <td>1132.462122</td>
      <td>10.386050</td>
      <td>2.135952</td>
      <td>2.003532</td>
      <td>1.153956</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.499900</td>
      <td>1.000000</td>
      <td>0.846154</td>
      <td>0.333333</td>
      <td>3.000000</td>
      <td>0.692308</td>
      <td>32.540000</td>
      <td>-124.350000</td>
      <td>0.149990</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.563400</td>
      <td>18.000000</td>
      <td>4.440716</td>
      <td>1.006079</td>
      <td>787.000000</td>
      <td>2.429741</td>
      <td>33.930000</td>
      <td>-121.800000</td>
      <td>1.196000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.534800</td>
      <td>29.000000</td>
      <td>5.229129</td>
      <td>1.048780</td>
      <td>1166.000000</td>
      <td>2.818116</td>
      <td>34.260000</td>
      <td>-118.490000</td>
      <td>1.797000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.743250</td>
      <td>37.000000</td>
      <td>6.052381</td>
      <td>1.099526</td>
      <td>1725.000000</td>
      <td>3.282261</td>
      <td>37.710000</td>
      <td>-118.010000</td>
      <td>2.647250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.000100</td>
      <td>52.000000</td>
      <td>141.909091</td>
      <td>34.066667</td>
      <td>35682.000000</td>
      <td>1243.333333</td>
      <td>41.950000</td>
      <td>-114.310000</td>
      <td>5.000010</td>
    </tr>
  </tbody>
</table>
</div>
```

## Data Visualization with seaborn

### Original Data

``` python
interest_attr = ["MedInc","AveRooms", "AveBedrms", "AveOccup", "Population", "MedHouseVal"]
subset = data[interest_attr]
subset.describe()
_ = sns.pairplot(data=subset, hue="MedHouseVal", palette="plasma", plot_kws=dict(linewidth=0))
```


![](/assets/img/scaling_python/69a6f974ca37adc30541dc8850b891a87015c1ae.png)



### Using midpoint with Quantize the target MedHouseVal


``` python
# Quantize the target MedHouseVal 
subset.loc[:,"MedHouseVal"] = pd.qcut(subset["MedHouseVal"], q = 6, precision=0)
# using midpoint
subset.loc[:,"MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)
_ = sns.pairplot(data=subset, hue="MedHouseVal", palette="plasma", plot_kws=dict(linewidth=0))
```


![](/assets/img/scaling_python/e19e2eb81c0b37f0c50868fcd8494a03b42874fb.png)


# Scaling Processing



## The Original Data with and without extreme values

I used \"Median income\" and \"Average house occupancy\" with mapping
color of target values Median House Values to demonstrate the affect of
scaling processing


``` python
values = ["Median income", "Median house age", "Average number of rooms", "Average number of bedrooms", 
           "Population", "Average house occupancy", "House latitude", "House longitude"]
keys = california_housing.feature_names
feature_mapping = dict(zip(keys, values))
#interest_attr = ["MedInc","AveRooms", "AveBedrms", "AveOccup", "Population", "MedHouseVal"]
make_plot(0, california_housing, feature_mapping, xvar = "MedInc", yvar = "AveOccup", color = "ocean_r")
```


![](/assets/img/scaling_python/40c171eedf681ee591bcee1e054aa1e22a1c77a5.png)



## 1. Standard Scaling (StandardScaler) 

Standardize features by removing the mean and scaling to unit variance.

The standard score of a sample x is calculated as:

z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False,
and s is the standard deviation of the training samples or one if
with_std=False.

Centering and scaling happen independently on each feature by computing
the relevant statistics on the samples in the training set. Mean and
standard deviation are then stored to be used on later data using
transform.

Standardization of a dataset is a common requirement for many machine
learning estimators: they might behave badly if the individual features
do not more or less look like standard normally distributed data (e.g.
Gaussian with 0 mean and unit variance).

For instance many elements used in the objective function of a learning
algorithm (such as the RBF kernel of Support Vector Machines or the L1
and L2 regularizers of linear models) assume that all features are
centered around 0 and have variance in the same order. If a feature has
a variance that is orders of magnitude larger than others, it might
dominate the objective function and make the estimator unable to learn
from other features correctly as expected.

``` python
make_plot(1, california_housing, feature_mapping, xvar = "MedInc", yvar = "AveOccup", color = "plasma_r")
```


![](/assets/img/scaling_python/fea0f10dc128176c712dfb1274e2c4c531eee7d5.png)

## 2. Min-Max Scaling (MinMaxScaler) 

If your data consists of attributes with different scales, many machine
learning algorithms can benefit from rescaling the attributes so that
they are all on the same scale.

Normalization using MinMaxScaler rescaled between 0 and 1 for the
attributes. This is useful for optimization algorithms used at the core
of machine learning algorithms such as gradient descent. It is also
useful for algorithms that weight inputs, such as regression and neural
networks, and algorithms that use distance measures, such as K nearest
neighbors.


``` python
make_plot(2, california_housing, feature_mapping, xvar = "MedInc", yvar = "Population", color = "ocean_r")
```


![](/assets/img/scaling_python/fef7fe2ab591b463a50372dc9f14ffde93960cea.png)


## 3. Max-Abs Scaling (MaxAbsScaler) 

Scale each feature by its maximum absolute value.

This estimator scales and translates each feature individually such that
the maximal absolute value of each feature in the training set will be
1.0. It does not shift/center the data, and thus does not destroy any
sparsity.


``` python
make_plot(2, california_housing, feature_mapping, xvar = "MedInc", yvar = "AveOccup", color = "plasma_r")
```


![](/assets/img/scaling_python/b78717227d9fe1f08ee6507c54c72d3a1651d079.png)

## 4. Robust Scaling (RobustScaler) 

Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to the
quantile range (defaults to IQR: Interquartile Range). The IQR is the
range between the 1st quartile (25th quantile) and the 3rd quartile
(75th quantile).

Centering and scaling happen independently on each feature by computing
the relevant statistics on the samples in the training set. Median and
interquartile range are then stored to be used on later data using the
transform method.

Standardization of a dataset is a common requirement for many machine
learning estimators. Typically this is done by removing the mean and
scaling to unit variance. However, outliers can often influence the
sample mean / variance in a negative way. In such cases, the median and
the interquartile range often give better results.

``` python
make_plot(4, california_housing, feature_mapping, xvar = "MedInc", yvar = "Population", color = "summer_r")
```


![](/assets/img/scaling_python/5e0146d6f468996902613a896ab5562d5dfa3567.png)

## 5. Power Transformation (Yeo-Johnson) 

Apply a power transform featurewise to make data more Gaussian-like.

Power transforms are a family of parametric, monotonic transformations
that are applied to make data more Gaussian-like. This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

Currently, PowerTransformer supports the Box-Cox transform and the
Yeo-Johnson transform. The optimal parameter for stabilizing variance
and minimizing skewness is estimated through maximum likelihood.

Box-Cox requires input data to be strictly positive, while Yeo-Johnson
supports both positive or negative data.

By default, zero-mean, unit-variance normalization is applied to the
transformed data.

``` python
make_plot(5, california_housing, feature_mapping, xvar = "MedInc", yvar = "Population", color = "ocean_r")
```


![](/assets/img/scaling_python/5a46ca5b724c9b0b90337bcb43b99531a03859e0.png)


## 6. Power Transformation (Box-Cox) 


``` python
make_plot(6, california_housing, feature_mapping, xvar = "MedInc", yvar = "Population", color = "ocean_r")
```


![](/assets/img/scaling_python/423c957cca84e6264f6c6eb1fe04465aa83e8a35.png)

## 7. Quantile Transformation (Uniform pdf) 

Transform features using quantiles information.

This method transforms the features to follow a uniform or a normal
distribution. Therefore, for a given feature, this transformation tends
to spread out the most frequent values. It also reduces the impact of
(marginal) outliers: this is therefore a robust preprocessing scheme.

The transformation is applied on each feature independently. First an
estimate of the cumulative distribution function of a feature is used to
map the original values to a uniform distribution. The obtained values
are then mapped to the desired output distribution using the associated
quantile function. Features values of new/unseen data that fall below or
above the fitted range will be mapped to the bounds of the output
distribution. Note that this transform is non-linear. It may distort
linear correlations between variables measured at the same scale but
renders variables measured at different scales more directly comparable.


``` python
make_plot(7, california_housing, feature_mapping, xvar = "MedInc", yvar = "AveOccup", color = "plasma_r")
```


![](/assets/img/scaling_python/227bb0b87fce61a0ef52d115b161a390f740ee03.png)

## 8. Quantile Transformation (Gaussian pdf) 

``` python
make_plot(8, california_housing, feature_mapping, xvar = "MedInc", yvar = "AveOccup", color = "ocean_r")
```


![](/assets/img/scaling_python/799567ea5b6588b9690f15e18e565b560e45ae54.png)

## 9. Sample-wise L2 Normalizing (Normalizer) 

Normalize samples individually to unit norm.

Each sample (i.e. each row of the data matrix) with at least one non
zero component is rescaled independently of other samples so that its
norm (l1, l2 or inf) equals one.

This transformer is able to work both with dense numpy arrays and
scipy.sparse matrix (use CSR format if you want to avoid the burden of a
copy / conversion).

Scaling inputs to unit norms is a common operation for text
classification or clustering for instance. For instance the dot product
of two l2-normalized TF-IDF vectors is the cosine similarity of the
vectors and is the base similarity metric for the Vector Space Model
commonly used by the Information Retrieval community.

``` python
make_plot(9, california_housing, feature_mapping, xvar = "MedInc", yvar = "AveOccup",  color = "ocean_r")
```

![](/assets/img/scaling_python/5c184a2cf60f6d74b2d185b3f49cf2bd74133506.png)

