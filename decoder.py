import numpy as np
import matplotlib.pyplot as plt
import read_pfm
import scipy.io
import pickle
import os
import time

# Compute the time for encoding 
decoding_time = time.process_time()

#_______________________________________________________________________________________________________________________
# Load encoded data

# Specify folder paths
data_dir = './data/'
quantization_data = os.path.join(data_dir, 'quantization')
bitstream_data = os.path.join(data_dir, 'bitstream')

# Load residuals
with open(os.path.join(quantization_data, 'residuals.dat'), 'rb') as f:
    residuals = pickle.load(f)

with open(os.path.join(quantization_data, 'residual_img.pkl'), 'rb') as f:
    residual_img = pickle.load(f)

with open(os.path.join(quantization_data, 'residual_img_original_size.npy'), 'r') as file:
    residual_img_original_size = int(file.read())
    
dimensions_residual_img = np.fromfile(os.path.join(data_dir, 'dimensions_residual_img.dat'), dtype=int)

# Load encoded data from quantization

with open(os.path.join(quantization_data, 'compressed_size_fixed_rate.txt'), 'r') as file:
    compressed_size_fixed_rate = int(file.read())

with open(os.path.join(quantization_data, 'compressed_size_embedded.txt'), 'r') as file:
    compressed_size_embedded = int(file.read())

encoded_array_fixed_rate = np.load(os.path.join(data_dir, 'tagsBytesArray_fixed_rate.npy'))
encoded_array_embedded = np.load(os.path.join(data_dir, 'tagsBytesArray_embedded.npy'))

sortedProbability_fixed_rate_items = np.load(os.path.join(quantization_data, 'sortedProbability_fixed_rate.npy'), allow_pickle=True)
sortedProbability_fixed_rate = dict(sortedProbability_fixed_rate_items)

sortedProbability_embedded_items = np.load(os.path.join(quantization_data, 'sortedProbability_embedded.npy'), allow_pickle=True)
sortedProbability_embedded = dict(sortedProbability_embedded_items)

delta_fixed_rate = np.load(os.path.join(quantization_data, 'delta_fixed_rate.npy'))
delta_embedded = np.load(os.path.join(quantization_data, 'delta_embedded.npy'))
epsilon = np.load(os.path.join(quantization_data, 'epsilon.npy'))
maxad = np.load(os.path.join(quantization_data, 'maxad.npy'))

# Load segm_clothes encoded data

segm_clothes_original = np.load(os.path.join(data_dir, 'segm_clothes.npy'))
segm_clothes_shape = np.load(os.path.join(data_dir, 'segm_clothes_shape.npy'))

with open(os.path.join(quantization_data, 'segm_clothes_mean_values.dat'), 'rb') as f:
    segm_clothes_mean_values = pickle.load(f)

dimensions_segm_clothes = np.fromfile(os.path.join(data_dir, 'dimensions_segm_clothes.dat'), dtype=int)
encoded_array_segm_clothes = np.load(os.path.join(data_dir, 'tags_segm_clothes.npy'))

with open(os.path.join(data_dir, 'compressed_size_segm_clothes.npy'), 'rb') as file:
    compressed_size_segm_clothes = np.load(file)

with open(os.path.join(data_dir, 'colors.dat'), 'rb') as f:
    colors = pickle.load(f)

# Load data from arithmetic encoding 
with open(os.path.join(quantization_data, 'tags_fixed_rate.dat'), 'rb') as f:
    tagsBytesArray_fixed_rate = pickle.load(f)

probabilityVector = np.load(os.path.join(data_dir, 'probability_vector.npy'))

with open(os.path.join(data_dir, 'tags_segm_clothes.dat'), 'rb') as f:
    tags_bytes_array = pickle.load(f)

with open(os.path.join(quantization_data, 'segm_image_mean_values.dat'), 'rb') as f:
    segm_img_mean_values = pickle.load(f)

with open(os.path.join(data_dir,'segm_clothes_probs.npy'), 'rb') as f:
    segm_clothes_probs = pickle.load(f)

with open(os.path.join(quantization_data, 'segm_image_mean_values.dat'), 'rb') as f:
    segm_img_mean_values = pickle.load(f)

blockSize = 8

#_______________________________________________________________________________________________________________________
# Load dataset

normal_img = read_pfm.read_pfm('/Users/dianatat/Documents/licenta/datasets_vub/pfm_files/bigroom_180_normal_0.pfm')
segm_img = read_pfm.read_pfm('/Users/dianatat/Documents/licenta/datasets_vub/pfm_files/bigroom_180_segments_0.pfm')
mat = scipy.io.loadmat('/Users/dianatat/Documents/licenta/datasets_vub/mat_files/bigroom_180_segments_0.mat')
label_map = mat['label_map']

#_______________________________________________________________________________________________________________________
# Arithmetic decoder 

def bytes_to_bitstring(bytes_arr):
    return ''.join(format(byte, '08b') for byte in bytes_arr)

def convertBitStringToDecimal(bitString):
    num = 0
    idx = 1
    while idx < len(bitString) + 1:
        num += 2 ** (-idx) * (bitString[idx - 1] == '1')
        idx += 1
    return num

def arithmetic_decoder(tags, probability):
    numSymbols = len(probability)

    outputSymbols = []

    cumulative_sum = [0.0] * (numSymbols + 1)

    for i in range(1, numSymbols + 1):
        cumulative_sum[i] = cumulative_sum[i - 1] + probability[i - 1]  # 1 based cumulative sum

    cumulative_sum[numSymbols] = 1.0

    for tag in tags:
        l = 0.0
        u = 1.0
        iterations = 0
        while len(tag) > 0:
            if iterations >= 1000000:
                print("Max iteration limit reached. Exiting.")
                break

            if l >= 0.0 and u < 0.5:
                l = 2 * l
                u = 2 * u
                tag = tag[1:]
            elif l >= 0.5 and u <= 1.1:
                l = 2 * l - 1
                u = 2 * u - 1
                tag = tag[1:]
            elif l >= 0.25 and u < 0.75:
                l = 2 * l - 0.5
                u = 2 * u - 0.5
                if tag[0] == '0':
                    tag = tag[1:]
                else:
                    tag = tag[1:]
            else:
                t = convertBitStringToDecimal(tag)
                delta = u - l
                new_t = (t - l) / delta
                letterIdx = numSymbols - 1

                for i in range(len(cumulative_sum) - 1):
                    if new_t >= cumulative_sum[i] and new_t <= cumulative_sum[i + 1]:
                        letterIdx = i
                        break

                outputSymbols.append(letterIdx)
                new_l = l + delta * cumulative_sum[letterIdx]
                new_u = l + delta * cumulative_sum[letterIdx + 1]
                l = new_l
                u = new_u
                iterations += 1

    return outputSymbols

def decodeTags(tags):
    tagStringArray = []
    for tag in tags:
        tagStringArray.append(tag)
    return tagStringArray

#_______________________________________________________________________________________________________________________
# Read from the bistream

with open(os.path.join(bitstream_data, 'append_bitstream.bin'), 'rb') as f:
    loaded_bitstream = f.read()

def decode_bitstream(bitstream, segm_clothes_len, img_quantized_fixed_rate_len):
    # Convert bitstream to bitstring
    bitstring = bin(int.from_bytes(bitstream, 'big'))[2:]

    # Extract segm_clothes, img_quantized_fixed_rate, and img_quantized_embedded bitstrings
    segm_clothes_bitstring = bitstring[:segm_clothes_len]
    img_dequantized_fixed_rate_bitstring = bitstring[segm_clothes_len:segm_clothes_len+img_quantized_fixed_rate_len]
    img_dequantized_embedded_bitstring = bitstring[segm_clothes_len+img_quantized_fixed_rate_len:]

    # Decode each bitstring separately
    segm_clothes_decoded = arithmetic_decoder(segm_clothes_bitstring, segm_clothes_probs)
    img_dequantized_fixed_rate_decoded = arithmetic_decoder(img_dequantized_fixed_rate_bitstring, sortedProbability_fixed_rate)
    img_dequantized_embedded_decoded = arithmetic_decoder(img_dequantized_embedded_bitstring, sortedProbability_embedded)

    # Return decoded data
    return segm_clothes_decoded, img_dequantized_fixed_rate_decoded, img_dequantized_embedded_decoded

# Print the first 10 elements of the bitstream
print("First 10 elements of loaded bitstream: ", loaded_bitstream[:10])

# Print the length of the bitstream
print("Length of loaded bitstream: ", len(loaded_bitstream))

#_______________________________________________________________________________________________________________________
# Decompressing the residual image

tags_fixed_rate = decodeTags(encoded_array_fixed_rate)
tags_embedded = decodeTags(encoded_array_embedded)

probability = []
indices = {}

for i in range(len(probabilityVector)):
    if probabilityVector[i] > 0.0:
        indices[len(probability)] = i
        probability.append(probabilityVector[i])

original_stream_indices_fixed_rate = arithmetic_decoder(tags_fixed_rate, probability)
original_stream_indices_embedded = arithmetic_decoder(tags_embedded, probability)

original_stream_fixed_rate = []
original_stream_embedded = []

for idx in original_stream_indices_fixed_rate:
    original_stream_fixed_rate.append(indices[idx])

for idx in original_stream_indices_embedded:
    original_stream_embedded.append(indices[idx])

print('----------Checking the lenghts--------------')
print('Len of original_stream_fixed_rate',len(original_stream_fixed_rate))
print('Len of original_stream_embedded', len(original_stream_embedded))

# Truncate the original streams to fit the image dimensions
original_size = dimensions_residual_img[0] * dimensions_residual_img[1]
original_stream_fixed_rate = original_stream_fixed_rate[:original_size]
original_stream_embedded = original_stream_embedded[:original_size]

print('Len of truncated original_stream_fixed_rate', len(original_stream_fixed_rate))
print('Len of truncated original_stream_embedded', len(original_stream_embedded))

# Now reshape the streams
original_stream_fixed_rate_np = np.array(original_stream_fixed_rate).reshape(dimensions_residual_img)
original_stream_embedded_np = np.array(original_stream_embedded).reshape(dimensions_residual_img)

# Calculate the decompressed size based on the compressed size and compression ratio
decompressed_size_fixed_rate = dimensions_residual_img[0] * dimensions_residual_img[1] * residual_img.itemsize * 8
decompression_ratio_fixed_rate = decompressed_size_fixed_rate / compressed_size_fixed_rate
decompression_ratio_embedded = decompressed_size_fixed_rate / compressed_size_embedded

print('-----------Residual Image Decompression-----------')
print("Compressed size fixed_rate:", compressed_size_fixed_rate, "bits")
print("Compressed size embedded:", compressed_size_embedded, "bits")
print("Decompressed size:", decompressed_size_fixed_rate, "bits")
print("Decompression ratio:", decompression_ratio_fixed_rate)
print("Decompression ratio:", decompression_ratio_embedded)

#_______________________________________________________________________________________________________________________
# Dequantization of the residual image

def dequantizor(q, delta, n, epsilon):
    xi = epsilon * 2 ** (-n)
    deq = np.where(q != 0, np.sign(q) * (np.abs(q) - xi + delta) * delta, 0)
    return deq

# Reshape the streams back to the original image size
original_stream_fixed_rate_np = np.array(original_stream_fixed_rate)
original_stream_fixed_rate_truncated = original_stream_fixed_rate_np[:decompressed_size_fixed_rate]

original_stream_embedded_np = np.array(original_stream_embedded)
original_stream_embedded_truncated = original_stream_embedded_np[:decompressed_size_fixed_rate]

# Start with maxad = 1 and n = 0, and iterate until maxad equals 51
maxad = 1
n = 0

dequantized_image_fixed_rate = []
dequantized_image_embedded = []

# Start with maxad = 1 and n = 0, and iterate until maxad equals 51
maxad = 1
n = 0

while maxad <= 51:
    # Calculate delta values
    delta_fixed_rate = 2 * maxad
    delta_embedded = delta_fixed_rate / 3
    
    # Dequantize the images
    dequantized_image_fixed_rate = dequantizor(original_stream_fixed_rate_truncated, delta_fixed_rate, n, epsilon)
    dequantized_image_embedded = dequantizor(original_stream_embedded_truncated, delta_embedded, n, epsilon)

    # Increment maxad by 3 and n by 1
    maxad += 3
    n += 1

# Calculate the sizes after dequantization
size_after_dequantized_fixed_rate = dequantized_image_fixed_rate.size * dequantized_image_fixed_rate.itemsize * 8
size_after_dequantized_embedded = dequantized_image_embedded.size * dequantized_image_embedded.itemsize * 8
dequantization_ratio_fixed_rate = size_after_dequantized_fixed_rate / decompressed_size_fixed_rate
dequantization_ratio_embedded = size_after_dequantized_embedded / decompressed_size_fixed_rate

print('-----------Residual Image Dequantization-----------')
print(f"Size of images after dequantization (fixed rate): {size_after_dequantized_fixed_rate} bits")
print(f"Size of images after dequantization (embedded): {size_after_dequantized_embedded} bits")
print(f"Dequantization ratio (fixed rate): {dequantization_ratio_fixed_rate}")
print(f"Dequantization ratio (embedded): {dequantization_ratio_embedded}")

#_______________________________________________________________________________________________________________________
# Reconstruct residual image

def reverse_residual_compression(residual_img, label_map, segm_img_mean_values, segm_clothes_mean_values):

    # Create an empty image for the reconstructed residual image
    reconstructed_residual_img = np.zeros_like(residual_img)

    # Calculate the mean pixel values for each label in the label map
    for label, mean_value in segm_img_mean_values.items():
        residual = mean_value - segm_clothes_mean_values.get(label, 0)
        reconstructed_residual_img[label_map == label] = residual
        print(f"Residual for label {label}: {round(residual,2)}")

    return reconstructed_residual_img

# Call the reverse_residual_compression function
reconstructed_residual_img = reverse_residual_compression(residual_img, label_map, segm_img_mean_values, segm_clothes_mean_values)

#_______________________________________________________________________________________________________________________
# Decompress segm_clothes

tags_segm_clotes = decodeTags(encoded_array_segm_clothes)

probability = []
indices = {}

for i in range(len(probabilityVector)):
    if probabilityVector[i] > 0.0:
        indices[len(probability)] = i
        probability.append(probabilityVector[i])

original_stream_indices_segm_clothes= arithmetic_decoder(tags_segm_clotes, probability)

decoded_stream_segm_clothes = []

for idx in original_stream_indices_segm_clothes:
    decoded_stream_segm_clothes.append(indices[idx])

original_stream_indices_segm_clothes = decoded_stream_segm_clothes  # reassign the decoded stream

# Truncate the original streams to fit the image dimensions
original_size = dimensions_segm_clothes[0] * dimensions_segm_clothes[1]
original_stream_indices_segm_clothes = original_stream_indices_segm_clothes[:original_size]

# Now reshape the streams
original_stream_indices_segm_clothes = np.array(original_stream_indices_segm_clothes).reshape(dimensions_segm_clothes)

# Calculate the decompressed size based on the compressed size and compression ratio
decompressed_size_segm_clothes = dimensions_segm_clothes[0] * dimensions_segm_clothes[1] * 8
decompression_ratio_segm_clothes = decompressed_size_segm_clothes / decompressed_size_segm_clothes

print('-----------Segm_clothes Decompression-----------')
print("Compressed size:", compressed_size_segm_clothes, "bits")
print("Decompressed size:", decompressed_size_segm_clothes, "bits")
print("Decompression ratio:", decompression_ratio_segm_clothes)

#_______________________________________________________________________________________________________________________
# Reconstruct segm_clothes

def reverse_clothes_segmentation(segm_clothes, label_map):
    print('-----------Reverse Clothes Segmentation-----------')

    # Get the unique segments from the label map
    segments = np.unique(label_map)
    print(f"Unique segments: {segments}")

    # Create an empty image for the reconstructed normal image
    reconstructed_segm_clothes = np.zeros_like(segm_clothes)

    # Loop over each segment
    for segment in segments:
        if segment == 0:
            continue  # Ignore the background

        # Create a mask for the current segment in the label map
        mask = (label_map == segment)

        # Get the mean pixel value for the segment from the corresponding region in segm_clothes
        mean_pixel_value = np.mean(segm_clothes[mask])

        print(f"Mean pixel value of the segment {segment}: {mean_pixel_value}")

        # Apply the mean pixel value to the corresponding region in the reconstructed normal image
        reconstructed_segm_clothes[mask] = mean_pixel_value

    return reconstructed_segm_clothes

# Call the reverse_clothes_segmentation function
reconstructed_segm_clothes = reverse_clothes_segmentation(segm_clothes_original, label_map)
reconstructed_segm_clothes = reconstructed_segm_clothes.astype(np.uint8)

# Get unique labels and their frequencies
unique_labels, label_counts = np.unique(label_map, return_counts=True)
label_frequencies = dict(zip(unique_labels, label_counts))

# Calculate the range of pixel values for each clothing item
pixel_ranges = {}
for label in unique_labels:
    if label != 0:  # Exclude the background label
        pixels = label_map[label_map == label]
        pixel_range = (np.min(pixels), np.max(pixels))
        pixel_ranges[label] = pixel_range

# Define the background color
background_color = (255, 255, 255)  # white background

# Define the colors for each clothing label (change as desired)
colors = {
    0: (255, 255, 255),  # background - white
    1: (0, 0, 0), 
    4: (0, 255, 255),  
    5: (255, 0, 200),  
    7: (130, 255, 130)  
}

# Create an empty output image with the same shape as the label map
color_visualition_clothes = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)

# Assign colors to each clothing label in the output image
for label in unique_labels:
    if label in colors:
        clothing_mask = label_map == label
        color_visualition_clothes[clothing_mask] = colors[label]

#_______________________________________________________________________________________________________________________
# Reconstruct segmented image

def reconstruct_segmented_image(reconstructed_residual_img, reconstructed_segm_clothes, label_map):

    # Initialize the reconstructed original image
    reconstructed_segm_original_image_1 = np.zeros_like(reconstructed_residual_img)

    # Get the unique labels from the label map
    unique_labels = np.unique(label_map)

    # For each unique label, add the corresponding segments from the residual image and the segm_clothes
    for label in unique_labels:
        mask = (label_map == label)
        reconstructed_segm_original_image_1[mask] = reconstructed_residual_img[mask] + reconstructed_segm_clothes[mask]

    return reconstructed_segm_original_image_1

# Call the reconstruct_original_image function
reconstructed_segm_original_image_1 = reconstruct_segmented_image(reconstructed_residual_img, reconstructed_segm_clothes, label_map)
reconstructed_segm_original_image_1 = reconstructed_segm_original_image_1.astype(np.uint8)

#_______________________________________________________________________________________________________________________
# Reconstruct original image

def remove_segmentation(normal_img, reconstructed_segm_original_image_1, label_map):
    print('-----------Remove Segmentation-----------')

    # Copy the original image
    reconstructed_normal_img = normal_img.copy()

    # Get the unique segments from the label map
    segments = np.unique(label_map)
    print(f"Unique segments: {segments}")

    # Loop over each segment
    for segment in segments:
        if segment != 0:  
            mask = (label_map == segment)
            # Replace the segmented part with the original
            reconstructed_normal_img[mask] = reconstructed_segm_original_image_1[mask]

    return reconstructed_normal_img

# Call the remove_segmentation function
reconstructed_normal_img = remove_segmentation(normal_img, reconstructed_segm_original_image_1, label_map)
reconstructed_normal_img = reconstructed_normal_img.astype(np.uint8)

#_______________________________________________________________________________________________________________________
# Plot the results

fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(reconstructed_residual_img)
ax[0, 0].set_title('Reconstructed Residual Image')
ax[0, 1].imshow(color_visualition_clothes)
ax[0, 1].set_title('Reconstructed Color Visualition Clothes')
ax[0, 2].axis('off')
ax[1, 0].imshow(reconstructed_segm_clothes)
ax[1, 0].set_title('Reconstructed Segmented Clothes')
ax[1, 1].imshow(reconstructed_segm_original_image_1)
ax[1, 1].set_title('Reconstructed Segmented Image')
ax[1, 2].imshow(reconstructed_normal_img)
ax[1, 2].set_title('Reconstructed Normal Image')
plt.show()

#_______________________________________________________________________________________________________________________

print('-----------Decoding time-----------')
print(f"Decoding time {str(time.process_time() - decoding_time) + ' s'}")