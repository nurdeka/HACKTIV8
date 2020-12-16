#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap


# In[2]:


file = 'aceh_20_SRF.2014110100.nc' # mention the path to the downloaded file
data = Dataset(file, mode='r') # read the data 
print(type(data)) # print the type of the data 
print(data.variables.keys()) # print the variables in the data


# In[3]:


data


# In[7]:


lats = data.variables['xlat'][:]  
longs = data.variables['xlon'][:]
time = data.variables['time'][:]

tp = data.variables['pr'][:]


# In[11]:


lats.shape


# In[ ]:




