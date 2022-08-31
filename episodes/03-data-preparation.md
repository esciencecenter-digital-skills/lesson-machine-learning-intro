---
title: "Preparing your data for machine learning"
teaching: 15
exercises: 5
---

:::::: questions

- How do I check for missing data?
- How can I scale the data?

::::::

:::::: objectives

- Know how to check for missing data
- Be able to scale your data using scikitlearn's scalers

::::::

## Missing data
We splitted our data into train and test, but did not make any other modifications.
To make our data fit for machine learning, we need to:

* Handle missing, corrupt or incorrect data
* Do feature normalization

Let's start by looking how much missing data we have:

```python
import pandas as pd
weather_train = pd.read_csv('data/weather_train.csv')
weather_train.isna().sum().sum()
```
```output
0
```

We have no missing values so we can continue.

It could also be that we have corrupt data, leading e.g. to outliers in the data set. The pair plot in the previous episode could have hinted to this. For this dataset, we don't need to do anything about outliers.

## Feature normalization
As we saw in the pairplot, the magnitudes of the different features are not directly comparable with each other. Some of the features are in mm, others in degrees celcius, and the scales are different.

Most Machine Learning algorithms regard all features together in one multi-dimensional space. To do calculations in this space that make sense, the features should be comparable to each other, e.g. they should be scaled. There are two options for scaling:

- Normalization (Min-Max scaling)
- Standardization (scale by mean and variance)

In this case, we choose min_max scaling, because we do not know much about the distribution of our features. If we know that (some) features have a normal distribution, it makes more sense to do standardization.


```python
import sklearn.preprocessing
min_max_scaler = sklearn.preprocessing.MinMaxScaler()

feature_names = weather_train.columns[:-1]

weather_train_scaled = weather_train.copy()
weather_train_scaled[feature_names] = min_max_scaler.fit_transform(weather_train[feature_names])
```

::::::::::::::::::challenge

## Exercise: Comparing distributions
Compare the distributions of the numerical features before and after scaling. What do you notice?

:::::::::::::::::::::::::::

Let's look at the statistics before scaling:
```python
weather_train.describe()
```

And after scaling:
```python
weather_train_scaled.describe()
```

We save the data to use in our next notebook


```python
weather_train_scaled.to_csv('data/weather_train_scaled.csv', index=False)
```

:::::: keypoints

- You can check for missing data using `pandas` `.isna()` method
- You can scale your data using a scaler from the `sklearn.preprocessing` module

::::::::::::::::
