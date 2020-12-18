#!/usr/bin/env python
# coding: utf-8

# In[39]:


from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap, cm, shiftgrid
import os


# In[40]:


file = 'aceh_20_SRF.2014110100.nc' # mention the path to the downloaded file
nc = Dataset(file, mode='r') # read the data 
print(type(nc)) # print the type of the data 
print(nc.variables.keys()) # print the variables in the data


# In[45]:


print(nc['ts'])


# In[44]:


u    = nc.variables['ua100m'][0,0,:,:]
v    = nc.variables['va100m'][0,0,:,:]
ps   = nc.variables['ps'][1,:,:]/100 # pascal to hpa

lonu = nc.variables['xlon'][:]
lon = lonu[1,:]
latv = nc.variables['xlat'][:]
lat = latv[:,1]
x, y = np.meshgrid(lon, lat)

minlon = min(lon)
minlat = min(lat)
maxlon = max(lon)
maxlat = max(lat)


# In[46]:


# Get dimensions assuming 3D: time, latitude, longitude
ps1  = nc.variables['ps']
ps2  = nc.variables['ts']
time_dim, lat_dim, lon_dim = ps1.get_dims()
time_var = nc.variables[time_dim.name]
times = num2date(time_var[:], time_var.units)
latitudes = nc.variables['xlat'][:,1]
longitudes = nc.variables['xlon'][:][1,:]


# In[47]:


times_grid, latitudes_grid, longitudes_grid = [
    x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing='ij')]

# eksekusi dengan pandas
df = pd.DataFrame({
    'time': [t.isoformat() for t in times_grid],
    'latitude': latitudes_grid,
    'longitude': longitudes_grid,
    'press': (ps1[:]/100).flatten(),
    'temp (C)': (ps2[:]-273.15).flatten()})


# In[48]:


df


# In[49]:


# simpan dengan format .csv
output_dir = './'
filename = os.path.join(output_dir, 'table.csv')
print(f'Writing data in tabular form to {filename} (this may take some time)...')


df.to_csv(filename, index=False)        #menyimpan dalam format .csv
print('Done')


# In[50]:


ps1.shape


# In[51]:


df.count()


# In[52]:


16*37*67


# In[56]:


t = df.groupby('time')['press'].plot()


# In[57]:


m = Basemap(projection='mill',
            llcrnrlon=minlon,
            llcrnrlat=minlat,
            urcrnrlon=maxlon,
            urcrnrlat=maxlat,
            resolution='i')

x,y= np.meshgrid(lon,lat) # for this dataset, longitude is 0 through 360, so you need to subtract 180 to properly display on map
xx,yy = m(x,y)


# In[10]:


plt.figure(figsize=(14,7))
m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
m.drawcounties() # you can even add counties (and other shapefiles!)

parallels = np.arange(0,10,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(93,103,5.) # make longitude lines every 5 degrees from 95W to 70W

m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
temp = m.contourf(xx,yy,ps)
cb = m.colorbar(temp,"bottom", size="5%", pad="2%")
plt.title('Title')
cb.set_label('Bar')
plt.show()


# In[13]:


# Plot2
uu = u
vv = v

fig, ax = plt.subplots(figsize=(12,9))
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='white',lake_color='black',zorder=0)
m.drawparallels(np.arange(-10,10,5), labels=[1,0,1,1], fontsize=5)
m.drawmeridians(np.arange(100,115,5), labels=[1,1,0,1], fontsize=5)


# Quiver Angin
X = np.arange(0, xx.shape[1], 8)
Y = np.arange(0, yy.shape[0], 8)
points = np.meshgrid(Y, X)
lats = lat
u, v, X, Y = m.transform_vector(uu, vv, lon, lats, 31, 21, returnxy=True, masked=True)
q = m.quiver(X, Y, u, v, pivot='middle', scale=10, scale_units='inches')
plt.quiverkey(q, X=0.1, Y=-0.13, U=5, label='Kec = 5 m $s^{-1}$', labelpos='E', fontproperties={'size':8})
plt.show() 


# In[58]:


# Plot3
uu = u
vv = v

fig, ax = plt.subplots(figsize=(12,9))
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='white',lake_color='black',zorder=0)
m.drawparallels(np.arange(-10,10,5), labels=[1,0,1,1], fontsize=5)
m.drawmeridians(np.arange(100,115,5), labels=[1,1,0,1], fontsize=5)

#plot tekanan
temp = m.contourf(xx,yy,ps)
cb = m.colorbar(temp,"bottom", size="5%", pad="2%")
plt.title('Title')
cb.set_label('Bar')

# Quiver angin
X = np.arange(0, xx.shape[1], 8)
Y = np.arange(0, yy.shape[0], 8)
points = np.meshgrid(Y, X)
lats = lat
u, v, X, Y = m.transform_vector(uu, vv, lon, lats, 31, 21, returnxy=True, masked=True)
q = m.quiver(X, Y, u, v, pivot='middle', scale=10, scale_units='inches')
plt.quiverkey(q, X=0.1, Y=-0.13, U=5, label='Kec = 5 m $s^{-1}$', labelpos='E', fontproperties={'size':8})
plt.show() 


# In[ ]:




