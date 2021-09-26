# privacy-preserver

This module provides a simple tool for anonymizing a dataset using PySpark. Given a `pandas.dataframe` with relevant metadata mondrian_privacy_preserver generates an anonymized `pandas.dataframe`. This provides following privacy preserving techniques for the anonymization.

- K Anonymity
- L Diversity
- T Closeness

## Demo

Jupyter notebook for each of the following modules are included.

- Mondrian Based Anonymity (Single User Anonymization included)
- Clustering Based Anonymity
- Differential Privacy

## Requirements

* Python

Python versions above Python 3.6 and below Python 3.8 are recommended. The module is developed and tested on: 
Python 3.7.7 and pip 20.0.2. (It is better to avoid Python 3.8 as it has some compatibility issues with Spark)

## Installation

### Using pip

Use `pip install privacy_preserver` to install library

### using source code

Clone the repository to your PC and run `pip install .` to build and install the package.

## Usage

Usage of each module is described in the relevant section.

### For Mondrian Anonymization and Clustering Anonymization

You'll need to construct a schema to get the anonymized `pandas.dataframe` dataframe.
You need to consider the column names and thier data types to construct this. Output of functions of the Mondrian and Clustering Anonymization is described in thier relevant sections. 

Following code snippet shows how to construct an example schema.

```python
import pandas as pd

#age, occupation - feature columns
#income - sensitive column

schema = [
    "age", 
    "occupation",
    "income",
]
```
_______________________________________________________________________________________________________

## Basic Mondrian Anoymizing


### Requirements - Basic Mondrian Anonymize

- Pandas. You can easily install it with `pip install pandas`.

### K Anonymity

The `pandas.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows.
Return type of all the non categorical columns will be string
You need to always consider the count column when constructing the schema. Count column is an integer type column.

```python
from privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - pandas.dataframe - original dataframe
#k - int - value of the k
#feature_columns - list - what you want in the output dataframe
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - schema of the output dataframe you are expecting

# df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
df = pd.read.csv(your_csv_file)['age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income']

categorical = set((
    'occupation',
    'sex',
    'race'
))

feature_columns = ['age', 'occupation']

sensitive_column = 'income'

your_anonymized_dataframe = Preserver.k_anonymize(df,
                                                k,
                                                feature_columns,
                                                sensitive_column,
                                                categorical,
                                                schema)
```

### K Anonymity (without row suppresion)

This function provides a simple way to anonymize a dataset which has an user identification attribute without grouping the rows.    
This function doesn't return a dataframe with the count variable as above function. Instead it returns the same dataframe, k-anonymized. Return type of all the non categorical columns will be string.   
User attribute column **must not** be given as a feature column and its return type will be same as the input type.   
Function takes exact same parameters as the above function. To use this method to anonymize the dataset, instead of calling `k_anonymize`, call `k_anonymize_w_user`.    

### L Diversity

Same as the K Anonymity, the `pandas.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows.
Return type of all the non categorical columns will be string
You need to always consider the count column when constructing the schema. Count column is an integer type column.

```python
from privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#l - int - value of the l
#feature_columns - list - what you want in the output dataframe
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting

df = pd.read.csv(your_csv_file)['age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income']

categorical = set((
    'occupation',
    'sex',
    'race'
))

feature_columns = ['age', 'occupation']

sensitive_column = 'income'

your_anonymized_dataframe = Preserver.l_diversity(df,
                                                k,
                                                l,
                                                feature_columns,
                                                sensitive_column,
                                                categorical,
                                                schema)
```

### L Diversity (without row suppresion)

This function provides a simple way to anonymize a dataset which has an user identification attribute without grouping the rows.   
This function doesn't return a dataframe with the count variable as above function. Instead it returns the same dataframe, l-diversity anonymized. Return type of all the non categorical columns will be string.    
User attribute column **must not** be given as a feature column and its return type will be same as the input type.   
Function takes exact same parameters as the above function. To use this method to anonymize the dataset, instead of calling `l_diversity`, call `l_diversity_w_user`.  

### T - Closeness

Same as the K Anonymity, the `pandas.sql.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows.
Return type of all the non categorical columns will be string
You need to always consider the count column when constructing the schema. Count column is an integer type column.

```python
from privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#l - int - value of the l
#feature_columns - list - what you want in the output dataframe
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting

df = spark.read.csv(your_csv_file).toDF('age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income')

categorical = set((
    'occupation',
    'sex',
    'race'
))

feature_columns = ['age', 'occupation']

sensitive_column = 'income'

your_anonymized_dataframe = Preserver.t_closeness(df,
                                                k,
                                                t,
                                                feature_columns,
                                                sensitive_column,
                                                categorical,
                                                schema)

```

### T Closeness (without row suppresion)

This function provides a simple way to anonymize a dataset which has an user identification attribute without grouping the rows.  
This function doesn't return a dataframe with the count variable as above function. Instead it returns the same dataframe, t-closeness anonymized. Return type of all the non categorical columns will be string.   
User attribute column **must not** be given as a feature column and its return type will be same as the input type.   
Function takes exact same parameters as the above function. To use this method to anonymize the dataset, instead of calling `t_closeness`, call `t_closeness_w_user`.  

### Single User K Anonymity

This function provides a simple way to anonymize a given user in a dataset. Even though this doesn't use the mondrian algorithm, function is included in the `mondrian_preserver`. User identification attribute and the column name of the user identification atribute is needed as parameters.   
This doesn't return a dataframe with count variable. Instead this returns the same dataframe, anonymized for the given user. Return type of user column and all the non categorical columns will be string.

```python
from privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#user - name, id, number of the user. Unique user identification attribute.
#usercolumn_name - name of the column containing unique user identification attribute.
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting
#random - a flag by default set to false. In a case where algorithm can't find similar rows for given user, if this is set to true, slgorithm will randomly select rows from dataframe.

df = spark.read.csv(your_csv_file).toDF('name',
    'age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income')

categorical = set((
    'occupation',
    'sex',
    'race'
))

sensitive_column = 'income'

user = 'Jon'

usercolumn_name = 'name'

random = True

your_anonymized_dataframe = Preserver.anonymize_user(df,
                                                k,
                                                user,
                                                usercolumn_name,
                                                sensitive_column,
                                                categorical,
                                                schema,
                                                random)

```
_______________________________________________________________________________________________________

## Introduction to Differential Privacy

Differential privacy is one of the data preservation paradigms similar to K-Anonymity, T-Closeness and L-Diversity.
It alters each value of actual data in a dataset according to specific constraints set by the owner and produces a 
differentially-private dataset. This anonymized dataset is then released for public utilization.

**ε-differential privacy** is one of the methods in differential privacy. Laplace based ε-differential privacy is 
applied in this library. The method states that the randomization should be according the **epsilon** (ε) 
(should be >0) value set by data owner. After randomization a typical noise is added to the dataset. It is calibrated 
according to the **sensitivity** value (λ) set by the data owner. 

Other than above parameters, a third parameter **delta** (δ) is added into the mix to increase accuracy of the 
algorithm. A **scale** is computed from the above three parameters and a new value is computed.

> scale = λ / (ε - _log_(1 - δ))

> random_number = _random_generator_(0, 1) - 0.5

> sign = _get_sign_(random_number)

> new_value = value - scale × sign × _log_(1 - 2 × _mod_(random_number))

In essence above steps mean that a laplace transform is applied to the value and a new value is randomly selected 
according to the parameters. When the scale becomes larger, the deviation from original value will increase.

TO BE DO

_______________________________________________________________________________________________________

## Clustering Anonymizer

### Requirements - Clustering Anonymize

* Pandas `pip intall pandas`
* kmodes `pip install kmodes`

### Clustering Based K Anonymity

Only recommend if there are more catogorical columns, than numerical column. if there are more numerical column, then modrian algorithm is recommended. 

It is recommended to use 5 <= k <= 20 to minimize the data loss, if your data set is small better to use a small k value 

he spark.sql.dataframe you get after anonymizing will always contain a extra column count which indicates the number of similar rows. Return type of all the non categorical columns will be string


In Clustering base Anonymizer you can choose how the how to initialize the cluster centroids. 

1. 'fcbg' = This method return cluster centroids weight on the probability of row's column values appear in dataframe. Default Value.
2. 'rsc'  = This method will choose centroids weight according to the column that has most number of unique values.
3. 'random = Return cluster centroids in randomly.

just enter the `center_type= 'fcbg'`to use fcbg, default is **fcbg**

You can also decide the clustering method.
1. default is special method 
2. kmodes method 

if you want to use default dont enter anything to attribute `mode=`, else if you want to use the kmodes method `mode= 'kmode'`
if you have a huge data amount default is recommended. 

you can also decide the return mode. If this value equal to `return_mode=''equal` ; K anonymization will done with equal member clusters. Default value is 'Not_Equal'
Not equal is often run fast, but could be data lossy. equal is vice versa. 

Below is a full example:
```python
from clustering_preserver import Kanonymizer
from gv import init
from anonymizer import Anonymizer

df = pd.read_csv("reduced_adult.csv")

schema = [
    "age",
    "workcalss",
    "education_num",
    "matrital_status",
    "occupation",
    "sex",
    "race",
    "native_country",
    "class",
]

QI = ['age', 'workcalss','education_num', 'matrital_status', 'occupation', 'race', 'sex', 'native_country']
SA = ['class']
CI = [1,3,4,5,6,7]

k_df = Anonymizer.k_anonymize(
    df, schema, QI, SA, CI, k=10, mode='', center_type='random', return_mode='Not_equal', iter=1)
k_df.show()
```

### Clustering based L-Diversity

This method is recommended only for k anonymized dataframe. 
Input anonymized dataframe will group into similar k clusters and clusters that have not L number of distinct sensitive attributes 
will be suspressed.
Recommended small number of l to minimum the data loss. Default value is l = 2.

```python
## k_df - K anonymized spark dataframe
## schema - output spark dataframe schema
## QI - Quasi Identifiers. Type list
## SA = Sensitive attributes . Type list

 QI = ['column1', 'column2', 'column3']
 CI = [1, 2]
 SA = ['column4']
 schema = [
     "column1",
     "column2",
     "column3",
     "column4",
 ]

l_df = Anonymizer.l_diverse(k_df,schema, QI,SA, l=2)
l_df.show()
```

### Clustering based T closeness

This method is recommended only for k anonymized dataframe. 
Input anonymized dataframe will group into similar k clusters and clusters that not have sensitive attribute distribution according to t value will be suspressed.
t should be in between 0 and 1.
Larger value of t to minimum the data loss. Default value is t = 0.2.

```python 
## k_df - K anonymized spark dataframe
## schema - output spark dataframe schema
## QI - Quasi Identifiers. Type list
## SA = Sensitive attributes . Type list

 QI = ['column1', 'column2', 'column3']
 CI = [1, 2]
 SA = ['column4']
 schema = [
     "column1",
     "column2",
     "column3",
     "column4",
 ]

t_df = Anonymizer.t_closer(
    k_df,schema, QI, SA, t=0.3, verbose=1)
t_df.show()
```
