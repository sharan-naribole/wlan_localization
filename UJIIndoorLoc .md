
# UJIIndoorLoc WLAN Fingerprint dataset EDA

In this notebook, I begin from the raw UJIIndoorLoc dataset and perform an initial exploratory data analysis on the distributions of the predictors and response variables. This is followed by dimensionality reduction analysis. Once the data is prepared, I focus on the model selection and finally ensemble learning.

## Table of Contents

* [Dataset Description](#dataset-description)

* [Data Pre-Processing](#preprocess)

* [Exploratory Data Analysis](#eda)

* [Skewness and Kurtosis](#skew-kurtosis)

* [Box-Cox Transformation](#box-cox)

* [Dimensionality Reduction](#dimension-reduction)

* [Predictor Correlations](#predictor-correlations)

* [Principal Component Analysis (PCA)](#pca)


```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels
import matplotlib.pyplot as plt
import matplotlib

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
```


```python
train_data = pd.read_csv("data/trainingData.csv")
test_data = pd.read_csv("data/validationData.csv")
```

# Dataset Description <a id='dataset-description'></a>

Source: https://www.kaggle.com/giantuji/UjiIndoorLoc

- **WAP001-WAP520**: Intensity value for **Wireless Access Point** (WAP). WAP will be the acronym used for rest of this notebook. Negative integer values from -104 to 0 and +100. **Censored data:** Positive value 100 used if WAP was not detected.

- **Longitude**: Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000

- **Latitude**: Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018.

- **Floor**: Altitude in floors inside the building. Integer values from 0 to 4.

- **BuildingID**: ID to identify the building. Measures were taken in three different buildings. Categorical integer values from 0 to 2.

- **SpaceID**: Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken. Categorical integer values.

- **RelativePosition**: Relative position with respect to the Space (1 - Inside, 2 - Outside in Front of the door). Categorical integer values.

- **UserID**: User identifier (see below). Categorical integer values.

- **PhoneID**: Android device identifier (see below). Categorical integer values.

- **Timestamp**: UNIX Time when the capture was taken. Integer value.


```python
train_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAP001</th>
      <th>WAP002</th>
      <th>WAP003</th>
      <th>WAP004</th>
      <th>WAP005</th>
      <th>WAP006</th>
      <th>WAP007</th>
      <th>WAP008</th>
      <th>WAP009</th>
      <th>WAP010</th>
      <th>...</th>
      <th>WAP520</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>FLOOR</th>
      <th>BUILDINGID</th>
      <th>SPACEID</th>
      <th>RELATIVEPOSITION</th>
      <th>USERID</th>
      <th>PHONEID</th>
      <th>TIMESTAMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>...</td>
      <td>100</td>
      <td>-7541.2643</td>
      <td>4.864921e+06</td>
      <td>2</td>
      <td>1</td>
      <td>106</td>
      <td>2</td>
      <td>2</td>
      <td>23</td>
      <td>1371713733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>...</td>
      <td>100</td>
      <td>-7536.6212</td>
      <td>4.864934e+06</td>
      <td>2</td>
      <td>1</td>
      <td>106</td>
      <td>2</td>
      <td>2</td>
      <td>23</td>
      <td>1371713691</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>-97</td>
      <td>100</td>
      <td>100</td>
      <td>...</td>
      <td>100</td>
      <td>-7519.1524</td>
      <td>4.864950e+06</td>
      <td>2</td>
      <td>1</td>
      <td>103</td>
      <td>2</td>
      <td>2</td>
      <td>23</td>
      <td>1371714095</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>...</td>
      <td>100</td>
      <td>-7524.5704</td>
      <td>4.864934e+06</td>
      <td>2</td>
      <td>1</td>
      <td>102</td>
      <td>2</td>
      <td>2</td>
      <td>23</td>
      <td>1371713807</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>...</td>
      <td>100</td>
      <td>-7632.1436</td>
      <td>4.864982e+06</td>
      <td>0</td>
      <td>0</td>
      <td>122</td>
      <td>2</td>
      <td>11</td>
      <td>13</td>
      <td>1369909710</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 529 columns</p>
</div>




```python
train_data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WAP001</th>
      <th>WAP002</th>
      <th>WAP003</th>
      <th>WAP004</th>
      <th>WAP005</th>
      <th>WAP006</th>
      <th>WAP007</th>
      <th>WAP008</th>
      <th>WAP009</th>
      <th>WAP010</th>
      <th>...</th>
      <th>WAP520</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>FLOOR</th>
      <th>BUILDINGID</th>
      <th>SPACEID</th>
      <th>RELATIVEPOSITION</th>
      <th>USERID</th>
      <th>PHONEID</th>
      <th>TIMESTAMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.0</td>
      <td>19937.0</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>...</td>
      <td>19937.0</td>
      <td>19937.000000</td>
      <td>1.993700e+04</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>19937.000000</td>
      <td>1.993700e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>99.823644</td>
      <td>99.820936</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>99.613733</td>
      <td>97.130461</td>
      <td>94.733661</td>
      <td>93.820234</td>
      <td>94.693936</td>
      <td>99.163766</td>
      <td>...</td>
      <td>100.0</td>
      <td>-7464.275947</td>
      <td>4.864871e+06</td>
      <td>1.674575</td>
      <td>1.212820</td>
      <td>148.429954</td>
      <td>1.833024</td>
      <td>9.068014</td>
      <td>13.021869</td>
      <td>1.371421e+09</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.866842</td>
      <td>5.798156</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.615657</td>
      <td>22.931890</td>
      <td>30.541335</td>
      <td>33.010404</td>
      <td>30.305084</td>
      <td>12.634045</td>
      <td>...</td>
      <td>0.0</td>
      <td>123.402010</td>
      <td>6.693318e+01</td>
      <td>1.223078</td>
      <td>0.833139</td>
      <td>58.342106</td>
      <td>0.372964</td>
      <td>4.988720</td>
      <td>5.362410</td>
      <td>5.572054e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-97.000000</td>
      <td>-90.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>-97.000000</td>
      <td>-98.000000</td>
      <td>-99.000000</td>
      <td>-98.000000</td>
      <td>-98.000000</td>
      <td>-99.000000</td>
      <td>...</td>
      <td>100.0</td>
      <td>-7691.338400</td>
      <td>4.864746e+06</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.369909e+09</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.0</td>
      <td>-7594.737000</td>
      <td>4.864821e+06</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>110.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>1.371056e+09</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.0</td>
      <td>-7423.060900</td>
      <td>4.864852e+06</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>129.000000</td>
      <td>2.000000</td>
      <td>11.000000</td>
      <td>13.000000</td>
      <td>1.371716e+09</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.0</td>
      <td>-7359.193000</td>
      <td>4.864930e+06</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>207.000000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>14.000000</td>
      <td>1.371721e+09</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>100.0</td>
      <td>-7300.818990</td>
      <td>4.865017e+06</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>254.000000</td>
      <td>2.000000</td>
      <td>18.000000</td>
      <td>24.000000</td>
      <td>1.371738e+09</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 529 columns</p>
</div>




```python
# Response variables in our problem are Building, Floor, Latitude, Longitude and Relative Position
(train_data[['FLOOR','BUILDINGID', 'SPACEID','RELATIVEPOSITION','USERID','PHONEID']]
.astype(str)
.describe(include=['object']))
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLOOR</th>
      <th>BUILDINGID</th>
      <th>SPACEID</th>
      <th>RELATIVEPOSITION</th>
      <th>USERID</th>
      <th>PHONEID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19937</td>
      <td>19937</td>
      <td>19937</td>
      <td>19937</td>
      <td>19937</td>
      <td>19937</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>3</td>
      <td>123</td>
      <td>2</td>
      <td>18</td>
      <td>16</td>
    </tr>
    <tr>
      <th>top</th>
      <td>3</td>
      <td>2</td>
      <td>202</td>
      <td>2</td>
      <td>11</td>
      <td>14</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5048</td>
      <td>9492</td>
      <td>484</td>
      <td>16608</td>
      <td>4516</td>
      <td>4835</td>
    </tr>
  </tbody>
</table>
</div>



From the [paper](http://ieeexplore.ieee.org/document/7275492/) on this dataset:
"Although both the training subset and the validation subset contain the same information, the latter includes the value 0 in some fields. These fields are: SpaceID, Relative Position with respect to SpaceID and UserID. As it has been commented before, this information was not recorded because the validation captures were taken at arbitrary points and the users were not tracked in this phase. **This fact tries to simulate a real localization system.**"

Hence, Space ID, Relative Position, User ID won't be used to model the Localization algorithm. Also, Phone iD won't be used as in a real system, new phones should be localized without being used in the training.

Next, I focus on the pre-processing of the WAP RSSI columns.

#  Data Pre-Processing <a id='preprocess'></a>

## Exploratory Data Analysis <a id='eda'></a>


```python
X_train = train_data.iloc[:,:520]
X_test = test_data.iloc[:,:520]

y_train = train_data.iloc[:,520:526]
y_test = test_data.iloc[:,520:526]
```


```python
X_train.shape
```




    (19937, 520)




```python
X_train = (X_train
             .replace(to_replace=100,value=np.nan))

# Perform the same transform on Test data
X_test = (X_test
             .replace(to_replace=100,value=np.nan))
```

We are replacing the out-of-range values with NaN to avoid disturbance to our analysis on in-range RSSI distribution.


```python
X_stack = X_train.stack(dropna=False)
sns.distplot(X_stack.dropna(),kde = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10d47b630>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_16_1.png)


Skewness is a measure of asymmetry of distribution. Clearly, the distribution above appears right-skewed with majority of the values being on the left side of the distribution. Let's look at the skewness value for inidividual WAP RSSI distributions! We might have to perform a log/ Box-Cox transformation to overcome the skewness. 

Let's look at percentage of out-of-range overall and column wise.


```python
# Proportion of out of range values
sum(X_stack.isnull() == 0)/len(X_stack)
```




    0.034605449473533938



**96.1% of the values in the matrix represent Out-of-Range.** This is expected as for any given measurement, only a subset of the APs might be in reach of the mobile device.

For this purpose, let's analyze the ditribution of number of APs in range for the training data samples.


```python
waps_in_range = (X_train
                 .notnull()
                 .sum(axis = 1))

fig, ax = plt.subplots(1,1)

sns.violinplot(waps_in_range, ax = ax)
ax.set_xlabel("Number of APs in range")
```




    <matplotlib.text.Text at 0x10d467a20>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_20_1.png)



```python
waps_in_range.describe()
```




    count    19937.000000
    mean        17.994834
    std          7.333575
    min          0.000000
    25%         13.000000
    50%         17.000000
    75%         22.000000
    max         51.000000
    dtype: float64



Interestingly, majority of the samples have over 13 APs in range with the maximum of 51 APs. We do observe some of the training samples with 0 APs in range. Let's remove these samples from the training data.


```python
print("Before sample removal:", len(X_train))

y_train = (y_train
          .loc[X_train
              .notnull()
              .any(axis=1),:])

X_train = (X_train
           .loc[X_train
                .notnull()
                .any(axis=1),:])


print("After sample removal:", len(X_train))
```

    Before sample removal: 19937
    After sample removal: 19861


We cannot delete training samples with just a single AP or few APs in range as that is the best information we have to localize. 

We can remove the RSSI columns related to APs which are not in range in any of our training samples.


```python
# Removing columns with all NaN values
all_nan = (X_train
           .isnull()
           .all(axis=0) == False)
filtered_cols = (all_nan[all_nan]
                 .index
                 .values)

print("Before removing predictors with no in-range values", X_train.shape)

X_train = X_train.loc[:,filtered_cols]
X_test = X_test.loc[:,filtered_cols]

print("After removing predictors with no in-range values", X_train.shape)
```

    Before removing predictors with no in-range values (19861, 520)
    After removing predictors with no in-range values (19861, 465)


## Skewness and Kurtosis <a id="skew-kurtosis"></a>

Skewness and kurtosis metrics are common measures to find out how close a distribution is to the normal distribution. When the data is far away from normality statistic significantly, Box-Cox transformation is one way to satisfy the normality. This is necessary for standard statistical tests, and also sometimes to satisfy the linear and/or the equal variance assumptions for a standard linear regression model


```python
# Finding skewness ignoring out-of-range values
X_skew = X_train.skew()

sns.distplot(X_skew.dropna(),kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10d966d68>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_28_1.png)


We can observe majority of the WAP columns have a low to medium positive skewness in the region (0,1). There are still a few columns outside the (1,-1) range typically considered an acceptable range of skewness.

Next, before we apply the Normality tests, we need to fill in the out-of-range values which are currently NaN. Box-Cox transformation requires all values to be positive. For this purpose, let's transform our predictors to normal scale from the dBm scale.

Also, the out-of-range values are transformed to 1/hundreth of the absolute minimum among all in-range values. Therefore, the transformed out-of-range value represents the minimum RSSI value in the dataset.


```python
X_exp_train = np.power(10,X_train/10,)
X_exp_test = np.power(10,X_test/10)

abs_min = (X_exp_train.apply(min).min())
print(abs_min)

X_exp_train.fillna(abs_min,inplace=True)
X_exp_test.fillna(abs_min,inplace=True)
```

    5.01187233627e-11



```python
plt.plot(X_exp_train.iloc[:,0])
```




    [<matplotlib.lines.Line2D at 0x10d890160>]




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_31_1.png)


### Normality test 

In this sub-section, I will explore various normality tests and explore the distributions of  RSSI predictors.

1. **Skew test**: Z-score of the test in which the null hypothesis states that that the skewness of the population that the sample was drawn from is the same as that of a corresponding normal distribution.

2. **Kurtosis test**: Z-score of the test in which the null hypothesis states that the kurtosis of the population from which the sample was drawn is that of the normal distribution: kurtosis = 3(n-1)/(n+1).

3. **k2**: $s^2 + k^2$, where $s$ is the z-score returned by skew test and $k$ is the z-score returned by kurtosistest.

4. **p_value**: A 2-sided chi squared probability for the hypothesis test that the sample comes from a normal distribution. Same test as k2. 


```python
from scipy.stats.mstats import normaltest, skewtest, kurtosistest, skew, kurtosis

def normal_test(s):
    s = s.dropna()
    
    # Minimum samples required for Kurtosis = 21
    # Minimum samples required for Skewness = 8
    if len(s) <=20:
        return [np.nan, np.nan]
    k2, pvalue = normaltest(s)
    return list(normaltest(s))

def skew_test(s):
    s = s.dropna()
    
    # Minimum samples required for Skewness = 8
    if len(s) <=8:
        return np.nan
    z_score,pval = skewtest(s)
    return z_score

def kurtosis_test(s):
    s = s.dropna()
    
    # Minimum samples required for Kurtosis = 21
    if len(s) <=20:
        return np.nan
    z_score,pval = kurtosistest(s)
    return z_score

def skew_score(s):
    s = s.dropna()
    return float(skew(s).data)

def kurtosis_score(s):
    s = s.dropna()
    return kurtosis(s)

def in_range(s):
    return (s > abs_min).sum()
```


```python
X_norm = pd.DataFrame({'Sample_Size': X_exp_train.apply(in_range),
                         #'Normality': X_train.apply(normal_test),
                         'Skewness': X_exp_train.apply(skew_score),
                         'Kurtosis': X_exp_train.apply(kurtosis_score),
                         #'Skew_Test': X_exp_train.apply(skew_test),
                         #'Kurtosis_Test': X_exp_train.apply(kurtosis_test)
                        })

'''
X_norm['k2'] = (X_norm['Normality']
                 .apply(lambda x: x[0]))

X_norm['p_value'] = (X_norm['Normality']
                 .apply(lambda x: x[1]))

X_norm.drop('Normality', axis = 1,inplace = True)
'''

X_norm.head(15)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Kurtosis</th>
      <th>Sample_Size</th>
      <th>Skewness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>WAP001</th>
      <td>1589.911556</td>
      <td>18</td>
      <td>38.296999</td>
    </tr>
    <tr>
      <th>WAP002</th>
      <td>1570.412878</td>
      <td>19</td>
      <td>38.774718</td>
    </tr>
    <tr>
      <th>WAP005</th>
      <td>988.020774</td>
      <td>40</td>
      <td>29.692772</td>
    </tr>
    <tr>
      <th>WAP006</th>
      <td>2649.644164</td>
      <td>308</td>
      <td>49.203758</td>
    </tr>
    <tr>
      <th>WAP007</th>
      <td>2544.345308</td>
      <td>578</td>
      <td>48.199343</td>
    </tr>
    <tr>
      <th>WAP008</th>
      <td>271.572833</td>
      <td>677</td>
      <td>15.905379</td>
    </tr>
    <tr>
      <th>WAP009</th>
      <td>1607.562517</td>
      <td>595</td>
      <td>36.943490</td>
    </tr>
    <tr>
      <th>WAP010</th>
      <td>1314.161272</td>
      <td>87</td>
      <td>33.682378</td>
    </tr>
    <tr>
      <th>WAP011</th>
      <td>8749.417194</td>
      <td>2956</td>
      <td>91.377383</td>
    </tr>
    <tr>
      <th>WAP012</th>
      <td>2186.418420</td>
      <td>2983</td>
      <td>46.709112</td>
    </tr>
    <tr>
      <th>WAP013</th>
      <td>2224.512227</td>
      <td>1975</td>
      <td>41.630716</td>
    </tr>
    <tr>
      <th>WAP014</th>
      <td>2208.568286</td>
      <td>1955</td>
      <td>41.864942</td>
    </tr>
    <tr>
      <th>WAP015</th>
      <td>254.567147</td>
      <td>1007</td>
      <td>14.216486</td>
    </tr>
    <tr>
      <th>WAP016</th>
      <td>501.190217</td>
      <td>999</td>
      <td>20.185881</td>
    </tr>
    <tr>
      <th>WAP017</th>
      <td>662.356727</td>
      <td>84</td>
      <td>24.103059</td>
    </tr>
  </tbody>
</table>
</div>




```python
(X_exp_train.iloc[:,0] > 0.0).sum()
```




    19861



Let's explore the relationship between Kurtosis scores and Skew scores.


```python
sns.jointplot(y="Kurtosis", x="Skewness", stat_func= None, data=X_norm)
```




    <seaborn.axisgrid.JointGrid at 0x10d8da320>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_38_1.png)


**Skewness:** For normally distributed data, the skewness should be about 0. A skewness value > 0 means that there is more weight in the left tail of the distribution. Similarly, a negative value indicates a left-skewed distribution with more weight on the right tail.

Clearly, many of the predictors have a skewness outside the expected range of 0,0

**Kurtosis:** Kurtosis is the fourth central moment divided by the square of the variance. If a distribution has  positive kurtosis, that means it has more in the tails than the normal distribution. Similarly, if a distribution has a negative kurtosis, it has less in the tails than the normal distribution. 

In the above figure, for the columns with a higher skewness score, the kurtosis is also more extreme.

The statistical significance of the Skewness and Kurtosis scores can be checked plotting the z-scores of the Skew test and Kurtosis tests.


```python
#sns.jointplot(y="Kurtosis_Test", x="Skew_Test", data=X_norm)
```

## Box-Cox Transformation <a id="box-cox"></a>

To apply the Box-Cox transform we have to first make all our data positive. As we performed the exponential transformation, our data is already positive.

For example, let's observe the how the Box-Cox transformation parameter $\lambda$ is fit for the first WAP RSSI predictor column in our current filtered training set. The figure below shows the Probability Plot Correlelation Coefficient, as obtained from probplot when fitting the Box-Cox transformed input predictor against a normal distribution.


```python
lmbdas, pppc = stats.boxcox_normplot(X_exp_train.iloc[:,100], -100, 100)

fig,ax = plt.subplots(1,1)
ax.plot(lmbdas,pppc,'bo')

_, maxlog = stats.boxcox(X_exp_train.iloc[:,0])
ax.axvline(maxlog, color='r')

ax.set_xlabel("$\lambda$")
ax.set_ylabel("Prob Plot Corr Coeff")
ax.text(x = -8, y = 0.1, s="$\lambda_{opt} = $" + str(maxlog))
```




    <matplotlib.text.Text at 0x1171014a8>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_44_1.png)


The maximum log likelihood function peaks at $\lambda = 1.1437$. Next, let's find out the $\lambda$s for different columns in our dataset.


```python
def box_cox_lambda(s):
    _, maxlog = stats.boxcox(s)
    return maxlog
```


```python
lambda_bc = X_exp_train.apply(box_cox_lambda)

X_boxcox_train = X_exp_train
X_boxcox_test = X_exp_test

for wap in X_boxcox_train:
    # Training data transform
    X_boxcox_train.loc[:,wap] = stats.boxcox(X_exp_train.loc[:,wap],lmbda = lambda_bc.loc[wap])
    # Test data transform
    X_boxcox_test.loc[:,wap] = stats.boxcox(X_exp_test.loc[:,wap],lmbda = lambda_bc.loc[wap])
```


```python
sns.distplot(lambda_bc, kde = False)
plt.title("Distribution of Box-Cox $\lambda$ across predictors")
plt.tight_layout()
```


![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_48_0.png)


The above figure shows the distribution of $\lambda$s that maximize log-likelihood function for each predictor. We can observe the two biggestbars are located at +5 and -2.5.


```python
# After Box-Cox
X_norm_post_boxcox = pd.DataFrame({'Skewness': X_boxcox_train.apply(skew_score),
                         'Kurtosis': X_boxcox_train.apply(kurtosis_score),
                         'BoxCox_Lambda': lambda_bc})

X_norm_post_boxcox.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BoxCox_Lambda</th>
      <th>Kurtosis</th>
      <th>Skewness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>WAP001</th>
      <td>5.636369</td>
      <td>-3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WAP002</th>
      <td>5.636369</td>
      <td>-3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WAP005</th>
      <td>5.636369</td>
      <td>-3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WAP006</th>
      <td>-15.078885</td>
      <td>--</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WAP007</th>
      <td>-6.989974</td>
      <td>29.3916</td>
      <td>5.602818</td>
    </tr>
    <tr>
      <th>WAP008</th>
      <td>-6.063101</td>
      <td>24.3721</td>
      <td>5.135374</td>
    </tr>
    <tr>
      <th>WAP009</th>
      <td>-5.750808</td>
      <td>28.4107</td>
      <td>5.514591</td>
    </tr>
    <tr>
      <th>WAP010</th>
      <td>5.636369</td>
      <td>-3</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WAP011</th>
      <td>-0.975030</td>
      <td>1.93886</td>
      <td>1.982143</td>
    </tr>
    <tr>
      <th>WAP012</th>
      <td>-0.968725</td>
      <td>1.87163</td>
      <td>1.965486</td>
    </tr>
  </tbody>
</table>
</div>



The kurtosis and skewness seems to have greatly reduced compared to before the Box-Cox transformation. Let's compare!


```python
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.scatter(y="Kurtosis", x="Skewness", data=X_norm)
ax1.set_xlabel("Skewness")
ax1.set_ylabel("Kurtosis")
ax1.set_title("Pre- Box-Cox")

ax2.scatter(y="Kurtosis", x="Skewness", data=X_norm_post_boxcox)
ax2.set_xlabel("Skewness")
ax2.set_ylabel("Kurtosis")
ax2.set_title("Post- Box-Cox")
```




    <matplotlib.text.Text at 0x10d2a5e48>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_52_1.png)


Note that the scales for the two figures above are two orders of magnitude lower for Kurtosis and one order of mangnitude lower for Skewness after the Box-Cox transformation.

I am interested in observing the univariate distribution of skewness and kurtosis to find out if majority of our predictors are now close to normal or not.


```python
sns.jointplot(y="Kurtosis", x="Skewness", stat_func = None,data=X_norm_post_boxcox)
```




    <seaborn.axisgrid.JointGrid at 0x10d89a470>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_54_1.png)


We can observe the biggest bars are located in the region [0,1) for skewness and [0,-3) for kurtosis. 

At this point, we can remove the predictors that have high skewness (> 3) or high kurtosis (> 1). However, as predictor corresponds to RSSI distribution of a single WAP, there might be samples for which one or more of these WAPs might be the closest and correspondingly have the highest measure. For this purpose, I do not perform any predictor selection at this stage.

Instead, we can explore how much of the variance in the dataset is explained by the predictors using Principal Component Analysis (PCA).

# Dimensionality Reduction <a id = "dimension-reduction"></a>

Dimensionality reduction is one of the key techniques to reduce the complexity. 

PCA is a simple dimensionality reduction technique that applies linear transformations on the original space. Among all the orthogonal linear projections, PCA minimizes the **reconstruction error**, which is the distance between the instance and its reconstruction from the lower-dimensional space. That is sum of the distances between points in original space and the corresponding points in lower-dimensional space.

Before we can perform the PCA analysis, we need to bring the predictors to the same scale. Then, we analyze the correlations between the predictors and remove highly correlated predictors. This is because adjoining nearly correlated variables increases the contribution of their common underlying factor to the PCA. We can remove highly correlated predictors algorithmically or removing the correlations by whitening the data (conversion to Identity Covariance Matrix).

## Feature Scaling

Most models require the predictors to be on the same scale for better performancee. The main exceptions are decision-tree based models which are not dependent on scaling as the splits are univariate.


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_std_train = sc.fit_transform(X_boxcox_train)
X_std_test = sc.transform(X_boxcox_test)

X_std_train = pd.DataFrame(X_std_train)
X_std_test = pd.DataFrame(X_std_test)
```


```python
X_std_train.shape, X_std_test.shape
```




    ((19861, 465), (1111, 465))




```python
all_zero= ((X_std_train == 0) 
           .all()==False)

all_zero[all_zero].index.values
```




    array([  4,   5,   6,   8,   9,  10,  11,  12,  13,  16,  17,  20,  21,
            22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,
            35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
            48,  49,  50,  51,  54,  55,  56,  57,  58,  59,  60,  61,  62,
            63,  64,  65,  66,  67,  70,  71,  72,  73,  74,  75,  77,  78,
            79,  80,  81,  82,  84,  85,  86,  87,  88,  89,  90,  91,  92,
            94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
           107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
           120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133,
           134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147,
           148, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
           163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
           176, 177, 178, 180, 181, 192, 193, 209, 210, 211, 212, 223, 224,
           228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 242, 248, 251,
           252, 253, 256, 258, 260, 262, 263, 267, 268, 278, 279, 280, 281,
           282, 283, 284, 285, 286, 291, 297, 300, 301, 302, 303, 305, 307,
           309, 311, 312, 313, 316, 317, 329, 330, 331, 332, 333, 334, 335,
           337, 338, 343, 349, 352, 353, 354, 357, 359, 361, 363, 364, 367,
           368, 388, 392, 404, 408, 429, 430, 432, 433, 434, 435, 436, 441,
           442, 446, 447, 448, 456, 461, 462])




```python
#After the Box-Cox transformation and scaling, few of the predictors are reduced to a constant value of 0
# Let's remove these predictors from the training and test data
all_zero= ((X_std_train == 0) 
           .all()==False)
filtered_cols = (all_zero[all_zero]
                 .index
                 .values)

print("Before removing predictors with only zeros", X_std_train.shape)

X_rm_train = X_std_train.loc[:,filtered_cols]
X_rm_test = X_std_test.loc[:,filtered_cols]

print("After removing predictors with only zeros", X_rm_train.shape)

```

    Before removing predictors with only zeros (19861, 465)
    After removing predictors with only zeros (19861, 254)


## Predictor Correlations <a id="predictor-correlations"></a>

[Read this explanation](http://stats.stackexchange.com/questions/50537/should-one-remove-highly-correlated-variables-before-doing-pca) about how PCA tends to over-emphasize the contributions of correlated predictors.


```python
X_train_corr = X_rm_train.corr()

fig = plt.figure(figsize=(15,15))
sns.heatmap(X_train_corr,xticklabels=False, yticklabels=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10d4546a0>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_66_1.png)


Clearly, we observe clusters of predictors that are highly correlated. Let's assign a threshold of 0.8 and see how many predictor pairs have correlation above this threshold.


```python
corr_stack = X_train_corr.stack()
corr_thresh = 0.9

# Total entries in correlation matrix above threshold
Nthresh = (abs(corr_stack) >= corr_thresh).sum()

# Subtracting the correlation of predictor with themselves which is equal to 1
Nthresh -= 254

# Pairwise correlations appear twice in the matrix
Nthresh *= 0.5

Nthresh
```




    16.0




```python
X_train_corr.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>16</th>
      <th>...</th>
      <th>435</th>
      <th>436</th>
      <th>441</th>
      <th>442</th>
      <th>446</th>
      <th>447</th>
      <th>448</th>
      <th>456</th>
      <th>461</th>
      <th>462</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>-0.032524</td>
      <td>0.092562</td>
      <td>-0.072297</td>
      <td>-0.072705</td>
      <td>0.306424</td>
      <td>0.326105</td>
      <td>-0.040012</td>
      <td>-0.039844</td>
      <td>-0.017040</td>
      <td>...</td>
      <td>-0.041436</td>
      <td>-0.064387</td>
      <td>-0.074897</td>
      <td>-0.096718</td>
      <td>-0.077951</td>
      <td>-0.095217</td>
      <td>-0.054997</td>
      <td>-0.052137</td>
      <td>-0.073452</td>
      <td>-0.093966</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.032524</td>
      <td>1.000000</td>
      <td>-0.033013</td>
      <td>-0.078445</td>
      <td>-0.078888</td>
      <td>-0.061610</td>
      <td>-0.060374</td>
      <td>-0.043415</td>
      <td>-0.043233</td>
      <td>-0.027544</td>
      <td>...</td>
      <td>-0.044960</td>
      <td>-0.069863</td>
      <td>-0.081267</td>
      <td>-0.104944</td>
      <td>-0.084581</td>
      <td>-0.103315</td>
      <td>-0.059674</td>
      <td>-0.056571</td>
      <td>-0.079698</td>
      <td>-0.101957</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.092562</td>
      <td>-0.033013</td>
      <td>1.000000</td>
      <td>-0.073385</td>
      <td>-0.073799</td>
      <td>0.424395</td>
      <td>0.410362</td>
      <td>-0.040614</td>
      <td>-0.040444</td>
      <td>0.367253</td>
      <td>...</td>
      <td>-0.042059</td>
      <td>-0.065356</td>
      <td>-0.076024</td>
      <td>-0.098174</td>
      <td>-0.079124</td>
      <td>-0.096650</td>
      <td>-0.055824</td>
      <td>-0.052921</td>
      <td>-0.074557</td>
      <td>-0.095380</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.072297</td>
      <td>-0.078445</td>
      <td>-0.073385</td>
      <td>1.000000</td>
      <td>0.860429</td>
      <td>-0.138749</td>
      <td>-0.137963</td>
      <td>-0.096506</td>
      <td>-0.096102</td>
      <td>-0.061228</td>
      <td>...</td>
      <td>0.059106</td>
      <td>0.013579</td>
      <td>-0.078226</td>
      <td>0.372824</td>
      <td>-0.069382</td>
      <td>0.302851</td>
      <td>0.244019</td>
      <td>0.085967</td>
      <td>-0.064203</td>
      <td>0.345796</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.072705</td>
      <td>-0.078888</td>
      <td>-0.073799</td>
      <td>0.860429</td>
      <td>1.000000</td>
      <td>-0.139531</td>
      <td>-0.138741</td>
      <td>-0.097050</td>
      <td>-0.096644</td>
      <td>-0.061573</td>
      <td>...</td>
      <td>0.059162</td>
      <td>0.020887</td>
      <td>-0.064465</td>
      <td>0.370518</td>
      <td>-0.066609</td>
      <td>0.301852</td>
      <td>0.255517</td>
      <td>0.080549</td>
      <td>-0.056104</td>
      <td>0.339997</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 254 columns</p>
</div>



Only 10 predictor correlation pairs have correlation above our defined threshold. As they are a small number compared to the total number of predictors, I do not remove any at this stage. In general, we can remove half of these predictors in the following manner:

1. Determine the two predictors A and B with largest absolute pairwise correlation.

2. Determine average correlation between A and other predictors. Repeat this for B.

3. If A has a larger average correlation, remove it. Otherwise, remove B.

4. Repeat 1-3 until no absolute correlation is above threshold.

I found this technique in the Chapter 3 of **Applied Predictive Modeling** book. I've written a few personal notes on the most important information I learnt reading this chapter. You can [find it here](https://github.com/sharan-naribole/applied-predictive-modeling/blob/master/Chapter-3.md).

## Principal Component Analysis (PCA) <a id = "pca"></a>

Dimensionality reduction is one of the key techniques to reduce the complexity. 

PCA is a simple dimensionality reduction technique that applies linear transformations on the original space. Among all the orthogonal linear projections, PCA minimizes the **reconstruction error**, which is the distance between the instance and its reconstruction from the lower-dimensional space. That is sum of the distances between points in original space and the corresponding points in lower-dimensional space.

An important point to remember about PCA is that it is an **unsupervised** form of dimensionality reduction. This means the response variables are not taken into consideration at any point of the transformation. sci-kit learn provides convenient methods to perform PCA which I'll be using directly.


```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_rm_train)
```




    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)




```python
# Borrowed from Sebastian Raschka's Python Machine Learning Book - Chapter 5

fig, ax = plt.subplots(1,1)

ax.bar(range(1, 255), pca.explained_variance_ratio_, alpha=0.5, align='center')
ax.step(range(1, 255), np.cumsum(pca.explained_variance_ratio_), where='mid')
ax.set_ylabel('Explained variance ratio')
ax.set_xlabel('Principal components')
ax.set_yticks(np.arange(0,1.1,0.1))
```




    [<matplotlib.axis.YTick at 0x116e9f240>,
     <matplotlib.axis.YTick at 0x1179a34a8>,
     <matplotlib.axis.YTick at 0x11703fb00>,
     <matplotlib.axis.YTick at 0x123538f98>,
     <matplotlib.axis.YTick at 0x12353cac8>,
     <matplotlib.axis.YTick at 0x1235405f8>,
     <matplotlib.axis.YTick at 0x123544128>,
     <matplotlib.axis.YTick at 0x123544c18>,
     <matplotlib.axis.YTick at 0x123549748>,
     <matplotlib.axis.YTick at 0x12354b278>,
     <matplotlib.axis.YTick at 0x12354bd68>]




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_74_1.png)


Roughly 95% of the variance is explained by first 150 eigen vectors. Before, we perform the dimensionality reduction on our data, let's analyze the reconstruction error as a function of the dimensions.


```python
X_rm_train = np.array(X_rm_train)
mu = np.mean(X_rm_train,axis = 0)

recon_error = []
for nComp in range(1,X_rm_train.shape[1]):
    #pca.components_ is already sorted by explained variance
    Xrecon = np.dot(pca.transform(X_rm_train)[:,:nComp], pca.components_[:nComp,:])
    Xrecon += mu
    recon_error.append(sum(np.ravel(np.abs(Xrecon- X_rm_train)**2)))
```


```python
pd.Series(recon_error).plot()
plt.xlabel("Number of Eigen Vectors")
plt.ylabel("Reconstruction Error")
```




    <matplotlib.text.Text at 0x1233ccdd8>




![png](UJIIndoorLoc%20_files/UJIIndoorLoc%20_77_1.png)


As the number of principal components used for the reconstruction increases, the reconstruction error expectedly decreases. This figure is a mirror image of the previous explained variance ratio figure. 

As 95% of the explained variance is explained by top 150 components, I will reduce my training and test data to 150 dimensions.


```python
Ndim_reduce = 150
X_train_pca = pca.transform(X_rm_train)[:,:Ndim_reduce]
X_test_pca = pca.transform(X_rm_test)[:,:Ndim_reduce]

X_train_pca.shape,X_test_pca.shape
```




    ((19861, 150), (1111, 150))




```python

```
