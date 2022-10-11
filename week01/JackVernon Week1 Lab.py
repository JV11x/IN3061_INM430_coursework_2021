# Email: jack.vernon@city.ac.uk
# Name: Jack Vernon

# QUESTION 1
# STEP 1: Select libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatpiial import distance
import csv
import sys

# %%

# STEP 2: Count Rows

f = open("TB_burden_countries_2014-09-29.csv")
for row in csv.reader(f):
    next(f)
    print(row)

data = pd.read_csv("TB_burden_countries_2014-09-29.csv")

print("Number of rows present = ",
      len(data))

# %%

# STEP 3:  Failed to complete after several attempts


# %%

# QUESTION 2
# STEP 1: Build Linspace and uniform arrays

x = np.linspace(5, 15, 6, dtype=int)
y = np.linspace(0, 23, 7)
z = np.random.uniform(-1, 1, 100)

# STEP 2: Print arrays, and ahow uniform array as a histogram

print(x)
print(y)
print(z)

plt.hist(z, 15, density=True)
plt.show

# %%

# STEP 3: Create 2 random arrays with 10 elments

w = np.random.random(10)
v = np.random.random(10)

print(w)
print(v)

# STEP 4 Calculating Euclidean Distance between arrays 'w' and 'v' using two methods

dist1 = distance.euclidean(w, v)

dist2 = np.sqrt(np.sum((w - v) ** 2))

print(dist1)
print(dist2)# QUESTION 1
# STEP 1: Select libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import csv
import sys

#%%

#STEP 2: Count Rows

f = open("TB_burden_countries_2014-09-29.csv")
for row in csv.reader(f):
    next(f)
    print(row)



data = pd.read_csv("TB_burden_countries_2014-09-29.csv")

print("Number of rows present = ",
      len(data))

#%%

#STEP 3:  Failed to complete after several attempts


#%%

# QUESTION 2
# STEP 1: Build Linspace and uniform arrays

x = np.linspace(5, 15, 6, dtype=int)
y = np.linspace(0, 23, 7)
z = np.random.uniform(-1,1,100)

# STEP 2: Print arrays, and ahow uniform array as a histogram

print(x)
print(y)
print(z)

plt.hist(z, 15, density=True)
plt.show

#%%

# STEP 3: Create 2 random arrays with 10 elments

w = np.random.random(10)
v = np.random.random(10)

print(w)
print(v)

# STEP 4 Calculating Euclidean Distance between arrays 'w' and 'v' using two methods

dist1 = distance.euclidean(w, v)

dist2 = np.sqrt(np.sum((w - v) ** 2))

print(dist1)
print(dist2)