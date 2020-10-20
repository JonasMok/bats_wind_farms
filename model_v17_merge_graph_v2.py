import pandas as pd
import numpy as np
import os
from collections import Iterable
import time
start = time.time()


def flatten(lis):
    '''convert nested list into one dimensional list'''
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def spectral_centroid(x, samplerate):
    ''' source: https://stackoverflow.com/questions/24354279/python-spectral-centroid-for-a-wav-file'''

    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean



dt = pd.read_csv('PIPI_control.csv')
dt2 = pd.read_csv('PIPI_ground_1.csv')
dt3 = pd.read_csv('PIPI_ground_4.csv')
dt4 = pd.read_csv('PIPI_ground_5.csv')
dt5 = pd.read_csv('PIPI_ground_6.csv')
dt6 = pd.read_csv('PIPI_ground_7.csv')
dt7 = pd.read_csv('PIPI_nacelle_1.csv')


dt_v1 = pd.concat([dt, dt2, dt3, dt4, dt5, dt6, dt7], ignore_index=True)

files = dt_v1['filename'].values.tolist()

#replace the end of file name from 'wav' to ''
files_2=[]
for i in files:
    new = i.replace('.wav','')
    files_2.append(new)

dt_v1['filename_2'] = files_2
dt_v1 = dt_v1.set_index('filename_2')
#cleaning the columns to prepare to merge
dt_v1 = dt_v1.drop(dt_v1.columns[[0,1,2]], axis=1)
#print(dt_v1.head())
#print(dt_v1.tail())
#print(dt_v1.shape)
#print(len(files_2))


#import file
pipi = pd.read_csv('PIPI_PIPY_folder.csv')
#list of name files
pipi = pipi.drop(pipi.columns[[0]], axis=1)
pipi = pipi.rename({'OUT FILE': 'filename_2'}, axis=1) #rename the name of the column
pipi = pipi.set_index('filename_2')
#print(pipi.head())

pipi_2 = pd.merge(dt_v1, pipi, left_index =True, right_index=True)

#print(pipi_2.head())
#print(pipi_2)
#print(pipi_2.shape)
dt4 = pipi_2[['spec_centroid','PULSES','PIPI','PIPY','ratio']]




from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3, random_state=0)

# Fit model to points
model.fit(dt4)


# Determine the cluster labels of new_points: labels
labels = model.predict(dt4)

print(type(labels))
folder = list(flatten(pipi_2[['FOLDER']].to_numpy()))


df_test = pd.DataFrame({'labels': labels, 'turbine': folder})

ct = pd.crosstab(df_test['labels'], df_test['turbine'])
print(ct)

#ct.to_csv('PIPI_total_table_v2.csv')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assign the columns of new_points: xs and ys
y = dt4['ratio']
x = dt4['PULSES']
z = dt4['spec_centroid']

# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")

# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.2)


# Creating color map
#my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(x, y, z, alpha = 0.8, c = labels, marker ='^')

plt.title("K-means scatter plot - 3 Clusters by ratio, matching and Spec. centroid")
ax.set_xlabel('X-axis - Pulses', fontweight ='bold')
ax.set_ylabel('Y-axis - Ratio (matching / Pulses)', fontweight ='bold')
ax.set_zlabel('Z-axis - Spectral Centroid', fontweight ='bold')
#fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
ax.legend(*sctt.legend_elements(),loc="lower right", title="Clusters")

# show plot
plt.show()



print("it took", time.time() - start, "seconds.")
