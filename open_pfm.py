from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import read_pfm

#bigroom , normal 
image1 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/bigroom_180_normal_0.pfm')
image2 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/bigroom_180_normal_3.pfm')
image3 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/bigroom_180_normal_8.pfm')

#bigroom , segmented
image4 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/bigroom_180_segments_0.pfm')
image5 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/bigroom_180_segments_3.pfm')
image6 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/bigroom_180_segments_8.pfm')

#livingroom , normal
image7 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/living_180_normal_1.pfm')
image8 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/living_180_normal_10.pfm')

#livingroom , segmented
image9 =  read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/living_180_segments_1.pfm')
image10 = read_pfm.read_pfm('/Users/Diana/Documents/licenta/coding/pfm_files/living_180_segments_10.pfm')

#plot the images 
images_bigroom = [image1, image2, image3, image4, image5, image6]
images_living = [image7, image8, image9, image10]

titles_bigroom = ['Bigroom_normal_0','Bigroom_normal_3','Bigroom_normal_8','Bigroom_segments_0','Bigroom_segments_3', 'Bigroom_segments_8' ]
titles_living = ['Livingroom_normal_1','Livingroom_normal_10','Livingroom_segments_1', 'Livingroom_segments_10']

fig, axarr = plt.subplots(2, 3)
for i, ax in enumerate(axarr.flat):
    ax.imshow(images_bigroom[i])
    ax.set_title(titles_bigroom[i])


fig, axarr = plt.subplots(2, 2)
for i, ax in enumerate(axarr.flat):
    ax.imshow(images_living[i])
    ax.set_title(titles_living[i])
plt.show()

#find out the maximum of for each image
images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10]

for image in images:
    max_val = np.max(image)
    shape = image.shape
    print(f"Maximum value for image: {max_val}")
    print(f"Shape of image: {shape}")



