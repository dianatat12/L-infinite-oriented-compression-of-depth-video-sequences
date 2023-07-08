import scipy.io
import matplotlib.pyplot as plt
import numpy as np

path = r'/Users/dianatat/Documents/licenta/datasets_vub/mat_files'

mat1 = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/bigroom_180_segments_0.mat') 
mat2 = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/bigroom_180_segments_3.mat') 
mat3 = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/bigroom_180_segments_8.mat') 
mat4 = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/living_180_segments_1.mat') 
mat5 = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/living_180_segments_1.mat') 

print(scipy.io.whosmat('/Users/Diana/Documents/licenta/coding/mat_files/bigroom_180_segments_0.mat')) #find out what variable has the mat file, what shape and what data type

mat = [mat1,mat2,mat3,mat4,mat5]
data = []

titles = ['Bigroom_segments_0','Bigroom_segments_3','Bigroom_segments_8','Livingroom_segments_1','Livingroom_segments_10']

for matrix in mat:
    data.append(np.array(matrix['label_map']))

data1, data2, data3, data4, data5 = data
data = np.array(data)

fig, axes = plt.subplots(nrows=1, ncols=5)
for i, data in enumerate([data1, data2, data3, data4, data5]):
    ax = axes[i]
    ax.imshow(data)
    ax.set_title(titles[i])
        
plt.show()
