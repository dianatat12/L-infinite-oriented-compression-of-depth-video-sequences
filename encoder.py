import numpy as np
import matplotlib.pyplot as plt
import read_pfm
import cv2
import scipy.io
import pickle
import os
import time

# Compute the time for encoding 
encoding_time = time.process_time()

#___________________________________________________________________________________________________________
# Step 1: Read the dataset

original_image_1 = read_pfm.read_pfm('/Users/dianatat/Documents/licenta/datasets_vub/pfm_files/bigroom_180_normal_0.pfm')
original_image_2 = read_pfm.read_pfm ('/Users/dianatat/Documents/licenta/datasets_vub/pfm_files/bigroom_180_normal_3.pfm')
original_image_3 = read_pfm.read_pfm ('/Users/dianatat/Documents/licenta/datasets_vub/pfm_files/bigroom_180_normal_8.pfm')

segm_original_image_1 = read_pfm.read_pfm('/Users/dianatat/Documents/licenta/datasets_vub/pfm_files/bigroom_180_segments_0.pfm')
mat_original_image_1 = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/bigroom_180_segments_0.mat')
label_map = mat_original_image_1['label_map']

#___________________________________________________________________________________________________________
# Step 2: Image Segmentation

# make a binary mask to take only the human
mask = cv2.inRange(segm_original_image_1, 3, 4)

#apply the mask to the normal image
segm_original_image_1 = cv2.bitwise_and(original_image_1, original_image_1, mask= mask)

def apply_mask(image, label_map):
    unique_labels = np.unique(label_map)
    label_mean_values = {}

    for label in unique_labels:
        mask = (label_map == label)
        item_pixels = image[mask]
        item_pixels = item_pixels.astype(float)
        item_pixels = item_pixels[item_pixels != 1e+10]

        if item_pixels.size > 0:
            mean_pixel_value = np.nanmean(item_pixels)
            label_mean_values[label] = mean_pixel_value

    return label_mean_values

print('-----------Human Segmentation-----------')

#Call the functions
segm_original_image_1_mean_values = apply_mask(segm_original_image_1, label_map)

def print_segment_mean_values(label_mean_values):
    for label, mean_value in label_mean_values.items():
        print(f"Mean pixel value of the segment {label}: {round(mean_value, 2)}")

segm_original_image_1_mean_values = apply_mask(segm_original_image_1, label_map)

print_segment_mean_values(segm_original_image_1_mean_values)

#___________________________________________________________________________________________________________
# Step 3: Clothes Segmentation
# Get unique clothing items from the label map

def clothes_segmentation(original_image_1, label_map):
    print('-----------Clothes Segmentation-----------')

    # Get the unique segments from the label map
    segments = np.unique(label_map)
    print(f"Unique segments: {segments}")

    # Define a new image for the segmented human with the mean pixel values
    segmented_clothes = np.zeros_like(original_image_1)

    # Loop over each segment
    for segment in segments:
        if segment == 0:
            continue # Ignore the background

        # Create a mask for the current segment
        mask = (label_map == segment)

        # Apply the mask to the normal image
        masked_img = original_image_1 * mask

        # Calculate the mean pixel value in the masked image
        segment_pixels = masked_img[mask]
        mean_pixel_value = np.mean(segment_pixels)

        print(f"Mean pixel value of the segment {segment}: {round(mean_pixel_value,2)}")

        # Apply the mean pixel value to the corresponding segment in the new image
        segmented_clothes[mask] = mean_pixel_value

    return segmented_clothes

segm_clothes = clothes_segmentation(original_image_1, label_map)
segm_clothes = segm_clothes.astype(np.uint8)

# Make a color visualization of the clothes

clothing_items = np.unique(label_map)

# Assign colors for each clothing label
colors = {
    0: (255, 255, 255),  # background - white
    1: (0, 0, 0),  # skin - black
    4: (0, 255, 0),  # t-shirt - green
    5: (255, 0, 0),  # shoes - blue
    7: (0, 0, 255),  # pants - yellow
}

def create_color_map(label_map, colors):
    color_map = np.zeros((*label_map.shape, 3), dtype=np.uint8)  # Initialize with zeros (black color)
    for label, color in colors.items():
        color_map[label_map == label] = color  # Assign color to corresponding pixels
    return color_map

# Call the functions
color_clothes = create_color_map(label_map, colors)

segm_clothes_mean_values = apply_mask(segm_clothes, label_map)

#___________________________________________________________________________________________________________
#Step 4: Apply Arithmetic Coding

def get_binaryStr_within_range(l, u):
    i = 1
    num = 0
    binaryStr = ''
    while not (num >= l and num < u):
        decimal_pnt = 2 ** (-i)
        new_decimal = num + decimal_pnt
        if new_decimal > u:
            binaryStr += '0'
        else:
            binaryStr += '1'
            num = new_decimal
        i += 1
    return binaryStr

def arith_encoding(img, block_size, sorted_probability):
    tags = []
    for i in range(len(img)):
        tag = ''
        l = 0.0
        u = 1.0
        value = int(img[i])
        for _ in range(block_size):
            new_l = l + (u - l) * sorted_probability.get(value, 0)
            new_u = l + (u - l) * sorted_probability.get(value + 1, 0)
            if new_l >= 0.5 and new_u < 0.5:
                l = 2 * l - 1
                u = 2 * u - 1
                tag += '1'
            elif new_l >= 0.25 and new_u < 0.75:
                l = 2 * l - 0.5
                u = 2 * u - 0.5
                tag += '0'
            elif new_l >= 0 and new_u < 0.5:
                l = 2 * l
                u = 2 * u
                tag += '0'
            elif new_l >= 0.5 and new_u <= 1.0:
                l = 2 * l - 1
                u = 2 * u - 1
                tag += '1'
            else:
                break
        tags.append(tag)
    return tags

def bitstring_to_bytes(s):
    # Pad s to make it divisible by 8
    num_bits = 8 - len(s) % 8
    s += '0' * num_bits
    return bytes(int(s[i: i + 8], 2) for i in range(0, len(s), 8))

blockSize = 8

# Normalize the segmented clothes image to be between 0 and 1
segm_clothes_normalized = (segm_clothes - np.min(segm_clothes)) / (np.max(segm_clothes) - np.min(segm_clothes))

# Scale the normalized image to be between 0 and 255
scaled_segm_clothes = (segm_clothes_normalized * 255).astype(int)

# Calculate pixel intensity histogram
hist, bin_edges = np.histogram(scaled_segm_clothes.ravel(), bins=range(257), density=True)

# Calculate cumulative distribution function
cdf = np.cumsum(hist)

# Map each pixel intensity to its cumulative probability
probs_limits = {i: (cdf[i - 1] if i > 0 else 0, cdf[i]) for i in range(256)}

# Step 4: Arithmetic Encoding
block_size = 8

# Flatten the grayscale image
img = scaled_segm_clothes.flatten()

# Compute the probability for each pixel value
probability = {}
for pix in img:
    if pix in probability.keys():
        probability[pix] += 1
    else:
        probability[pix] = 1

# Sort the probability dictionary
sorted_probability = {}
for shade in range(256):
    if shade in probability.keys():
        sorted_probability[shade] = probability[shade]

# Normalize the probabilities
total_pixels = len(img)
for p in sorted_probability:
    sorted_probability[p] /= total_pixels

# Apply arithmetic coding
tags_segm_clothes = arith_encoding(scaled_segm_clothes.flatten(), block_size, sorted_probability)

# Convert tags to bytes
tags_bytes_array = []
for tag in tags_segm_clothes:
    tags_bytes_array.append(bitstring_to_bytes(tag))

# Calculate the compressed size
original_size_segm_clothes = segm_clothes.size * 8
compressed_size_segm_clothes = sum(len(tag) for tag in tags_bytes_array)
compression_ratio_segm_clothes = original_size_segm_clothes / compressed_size_segm_clothes

print('-----------Segm_clothes compression-----------')
print("Original size of segm_clothes:", original_size_segm_clothes, "bits")
print("Compressed size of segm_clothes:", compressed_size_segm_clothes, "bits")
print("Compression ratio of segm_clothes:", compression_ratio_segm_clothes)
print("Shape of segm_clothes:", segm_clothes.shape)

#___________________________________________________________________________________________________________
# Step 5: Calculate residual image

print('-----------Residual Image-----------')

# Create an empty image to hold the residuals
residual_img = np.zeros_like(segm_original_image_1)

residuals = {label: segm_original_image_1_mean_values[label] - segm_clothes_mean_values.get(label, 0)
             for label in segm_original_image_1_mean_values.keys()}

# Print the results
for label, residual in residuals.items():
    print(f"Residual for label {label}: {round(residual,2)}")

# Create an empty image to hold the residuals
residual_img = np.zeros_like(segm_original_image_1)

# Assign residuals to corresponding segments
for label, residual in residuals.items():
    residual_img[label_map == label] = residual

residual_img_size_bits = residual_img.size * residual_img.itemsize * 8
print(f"Size of residual_img: {residual_img_size_bits} bits")

#___________________________________________________________________________________________________________
#plot the results

fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(original_image_1)
ax[0, 0].set_title('Original image')
ax[0, 1].imshow(segm_clothes)
ax[0, 1].set_title('Clothes segmented')
ax[0, 2].imshow(color_clothes)
ax[0, 2].set_title('Visual representation of the clothes')
ax[1, 0].imshow(segm_original_image_1)
ax[1, 0].set_title('Segmented image')
ax[1, 1].imshow(label_map)
ax[1, 1].set_title('Label map')
ax[1, 2].imshow(residual_img)
ax[1, 2].set_title('Residual image')
plt.show()

#___________________________________________________________________________________________________________
# Recreate the video sequence

# Load the frames
frames= [original_image_1, original_image_2, original_image_3]

# Normalize the frames
frames= [(image - image.min()) / (image.max() - image.min()) for image in frames]

# Convert to uint8 and scale from 0-255 for proper display
frames= [cv2.convertScaleAbs(image, alpha=(255.0)) for image in frames]

# Display images one after another in a loop
while True:
    for image in frames:
        cv2.imshow('Reconstructed video sequence', image)
        if cv2.waitKey(500) & 0xFF == ord('q'):  # Waits for 500ms, press 'q' to exit earlier
            break
    else:  
        continue
    break  

cv2.destroyAllWindows()

#___________________________________________________________________________________________________________
# Step 6: Calculate the fixed layer rate

# Calculate the layers
def quantizor_fixed_rate(x, delta_fixed_rate, epsilon):
    return np.where(np.abs(x) / delta_fixed_rate + epsilon > 0, np.round(np.sign(x) * (np.abs(x) / delta_fixed_rate + epsilon)), 0).astype(int)

def quantizor_embedded(x, delta_fixed_rate, epsilon, n):
    delta = delta_fixed_rate / 3
    return np.where(np.abs(x) / (3 ** n * delta) + epsilon / (3 ** n) > 0, np.round(np.sign(x) * (np.abs(x) / (3 ** n * delta) + epsilon / (3 ** n))), 0).astype(int)

def compute_probabilities(img, precision):
    img_rounded = np.round(img, decimals=precision)
    unique, counts = np.unique(img_rounded.astype(int), return_counts=True)
    prob = dict(zip(unique, counts / np.sum(counts)))
    return prob

# Set initial values
maxad = 51
epsilon = 0.5
n = 0

# Print initial values before quantization
print('-----------Quantization-----------')
print("Initial image sample values before quantization:")
print(residual_img[:3, :3])  # Print the first 3x3 values of the original image

sample_values_fixed_rate = []
sample_values_embedded = []

# Create list to store encoded bitstrings
all_encoded_bitstrings = []

# Convert segm_clothes encoded bitstream to single bitstring and add it to the list
all_encoded_bitstrings.append(''.join([format(byte, '08b') for byte in b''.join(tags_bytes_array)]))

# Continue with your while loop
while maxad >= 1:
    # Calculate delta values
    delta_fixed_rate = 2 * maxad
    delta_embedded = delta_fixed_rate / 3

    # Quantize the images
    img_quantized_fixed_rate = quantizor_fixed_rate(residual_img, delta_fixed_rate, epsilon)
    img_quantized_embedded = quantizor_embedded(residual_img, delta_fixed_rate, epsilon, n)

    # Calculate the probabilities and arithmetic coding for quantized image
    sortedProbability_fixed_rate = compute_probabilities(img_quantized_fixed_rate, precision=3)
    sortedProbability_embedded = compute_probabilities(img_quantized_embedded, precision=3)

    # Apply arithmetic coding
    tagsBytesArray_fixed_rate = arith_encoding(img_quantized_fixed_rate.flatten(), blockSize, sortedProbability_fixed_rate)
    tagsBytesArray_embedded = arith_encoding(img_quantized_embedded.flatten(), blockSize, sortedProbability_embedded)

    # Convert fixed rate encoded bitstream to single bitstring and add it to the list
    all_encoded_bitstrings.append(''.join(tagsBytesArray_fixed_rate))
    
    # Convert embedded encoded bitstream to single bitstring and add it to the list
    all_encoded_bitstrings.append(''.join(tagsBytesArray_embedded))

    # Decrement maxad by 3 and increment n by 1
    maxad -= 3
    n += 1

# Size before the quantization
residual_img_original_size = residual_img.size * residual_img.itemsize * 8
print(f"Original size of residual_img: {residual_img_original_size} bits")

# Calculate the sizes after quantization
size_after_fixed_rate = img_quantized_fixed_rate.size * img_quantized_fixed_rate.itemsize * 8
size_after_embedded = img_quantized_embedded.size * img_quantized_embedded.itemsize * 8

print(f"Size of residual_img after quantization (fixed rate): {size_after_fixed_rate} bits")
print(f"Size of residual_img after quantization (embedded): {size_after_embedded} bits")

# Calculate the probabilities and arithmetic coding for quantized image
sortedProbability_fixed_rate = compute_probabilities(img_quantized_fixed_rate, precision=3)
sortedProbability_embedded = compute_probabilities(img_quantized_embedded, precision=3)

# Apply arithmetic coding
tagsBytesArray_fixed_rate = arith_encoding(img_quantized_fixed_rate.flatten(), blockSize, sortedProbability_fixed_rate)
tagsBytesArray_embedded = arith_encoding(img_quantized_embedded.flatten(), blockSize, sortedProbability_embedded)

# Calculate and print compression ratios
compressed_size_fixed_rate = sum(len(tag) for tag in tagsBytesArray_fixed_rate) // 8
compressed_size_embedded = sum(len(tag) for tag in tagsBytesArray_embedded) // 8

compression_ratio_fixed_rate = (1 - compressed_size_fixed_rate / size_after_fixed_rate) * 100
compression_ratio_embedded = (1 - compressed_size_embedded / size_after_embedded) * 100

print(f"Compressed size using quantizor_fixed_rate: {compressed_size_fixed_rate} bits")
print(f"Compression ratio using quantizor_fixed_rate: {compression_ratio_fixed_rate} %")
print(f"Compressed size using quantizor_embedded: {compressed_size_embedded} bits")
print(f"Compression ratio using quantizor_embedded: {compression_ratio_embedded} %")

# Concatenate all bitstrings into a single bitstring
combined_bitstring = ''.join(all_encoded_bitstrings)

# Convert the combined bitstring to bytes
combined_bitstream = int(combined_bitstring, 2).to_bytes((len(combined_bitstring) + 7) // 8, 'big')

# Print the first 20 characters of the combined bitstream
print("Length of loaded bitstream: ", len(combined_bitstream))
print("Bitstream sample:", bin(int.from_bytes(combined_bitstream[:20], 'big'))[2:].zfill(8))

#___________________________________________________________________________________________________________
# Step 7: Save the encoded data

# Specify folder paths
data_dir = './data/'
quantization_data = os.path.join(data_dir, 'quantization')
bitstream_data = os.path.join(data_dir, 'bitstream')

# Save data from segm_img
np.save(os.path.join(data_dir, 'segm_original_image_1.npy'), segm_original_image_1)
np.save(os.path.join(data_dir, 'segm_clothes.npy'), segm_clothes)
np.save(os.path.join(data_dir, 'mask.npy'), mask)
with open(os.path.join(quantization_data, 'segm_image_mean_values.dat'), 'wb') as f:
    pickle.dump(segm_original_image_1_mean_values, f)

# Save data from segm_clothes
np.save(os.path.join(data_dir, 'segm_clothes.npy'), segm_clothes)
np.save(os.path.join(data_dir, 'segm_clothes_shape.npy'), segm_clothes.shape)
np.array([segm_clothes.shape[0], segm_clothes.shape[1]]).tofile(os.path.join(data_dir, 'dimensions_segm_clothes.dat'))

with open(os.path.join(data_dir, 'segm_clothes_mean_values.dat'), 'wb') as f:
    pickle.dump(segm_clothes_mean_values, f)

with open(os.path.join(data_dir, 'colors.dat'), 'wb') as f:
    pickle.dump(colors, f)

segm_clothes_compressed = np.array(tags_bytes_array)
np.save(os.path.join(data_dir, 'compressed_size_segm_clothes.npy'), compressed_size_segm_clothes)

# Save data from residual_img
with open(os.path.join(quantization_data, 'residuals.dat'), 'wb') as f:
    pickle.dump(residuals, f)
    
with open(os.path.join(quantization_data, 'residual_img.pkl'), 'wb') as f:
    pickle.dump(residual_img, f)

np.array([residual_img.shape[0], residual_img.shape[1]]).tofile(os.path.join(data_dir, 'dimensions_residual_img.dat'))
with open(os.path.join(quantization_data, 'residual_img_original_size.npy'), 'w') as file:
    file.write(str(residual_img_original_size))

# Save data from arith_encoding

with open(os.path.join(quantization_data, 'compressed_size_fixed_rate.txt'), 'w') as file:
    file.write(str(compressed_size_fixed_rate))
    
with open(os.path.join(quantization_data, 'compressed_size_embedded.txt'), 'w') as file:
    file.write(str(compressed_size_embedded))

# Save the encoded tags (for pfm)
with open(os.path.join(quantization_data, 'tags_fixed_rate.dat'), 'wb') as f:
    pickle.dump(tagsBytesArray_fixed_rate, f)

# Save the encoded tags (for embedded)
with open(os.path.join(quantization_data, 'tags_embedded.dat'), 'wb') as f:
    pickle.dump(tagsBytesArray_embedded, f)

with open(os.path.join(data_dir, 'tags_segm_clothes.dat'), 'wb') as f:
    pickle.dump(tags_bytes_array, f)

with open(os.path.join(bitstream_data, 'append_bitstream.bin'), 'wb') as file:
    file.write(combined_bitstream)

with open(os.path.join(data_dir,'segm_clothes_probs.npy'), 'wb') as f:
    pickle.dump(sorted_probability, f)

# Convert dictionaries to lists of items
sortedProbability_fixed_rate_items = list(sortedProbability_fixed_rate.items())
sortedProbability_embedded_items = list(sortedProbability_embedded.items())

probability_vector = np.array(list(sorted_probability.values()), dtype=float)
np.save(os.path.join(data_dir, 'probability_vector.npy'), probability_vector)

np.save(os.path.join(quantization_data, 'sortedProbability_fixed_rate.npy'), sortedProbability_fixed_rate_items)
np.save(os.path.join(quantization_data, 'sortedProbability_embedded.npy'), sortedProbability_embedded_items)
np.save(os.path.join(data_dir,'tags_segm_clothes.npy'), tags_segm_clothes)
np.save(os.path.join(data_dir, 'tagsBytesArray_fixed_rate.npy'), tagsBytesArray_fixed_rate)
np.save(os.path.join(data_dir, 'tagsBytesArray_embedded.npy'), tagsBytesArray_embedded)

# Save data from quantization
np.save(os.path.join(quantization_data, 'delta_fixed_rate.npy'), delta_fixed_rate)
np.save(os.path.join(quantization_data, 'delta_embedded.npy'), delta_embedded)
np.save(os.path.join(quantization_data, 'epsilon.npy'), epsilon)
np.save(os.path.join(quantization_data, 'maxad.npy'), maxad)
np.save(os.path.join(quantization_data, 'quantized_fixed_rate.npy'), img_quantized_fixed_rate)
np.save(os.path.join(quantization_data, 'quantized_embedded.npy'), img_quantized_embedded)
np.save(os.path.join(quantization_data, 'sample_values_fixed_rate.npy'), sample_values_fixed_rate)
np.save(os.path.join(quantization_data, 'sample_values_embedded.npy'), sample_values_embedded)

print('-----------Encoding time-----------')
print(f"Encoding time {str(time.process_time() - encoding_time) + ' s'}")