# L-infinite-oriented-compression-of-depth-video-sequences
L-infinite oriented compression of depth video sequences, performing arithmetic encoding, quantization fixed-rate and embedded, append to bitstream and decoder


## What is a Depth Camera?

A depth camera, also known as a 3D camera, is a type of imaging device that is capable of capturing not just the 2D representation of a scene (like a traditional camera), but also the depth information for each pixel. It achieves this by recording the distance between the camera and the objects in its field of view.

Depth cameras use various technologies to capture depth information. Some of the popular methods include time-of-flight, structured light, and stereo vision. The data captured by these cameras is often used in a range of applications, including 3D reconstruction, augmented reality, robotics, and gesture recognition. The depth data usually comes in the form of a depth map, which is a 2D image where each pixel's value represents its distance from the camera sensor.

Given the complexity and richness of the data they provide, working with depth cameras often involves specialized file formats. Two such formats we often use in this project are PFM and MAT, designed to handle high-precision floating-point values and complex structured data, respectively.
  
# Read depth camera pictures

PFM and MAT are common formats used for working with depth camera data. We use specific functions to read these files as they cannot be directly opened.

## What are PFM and MAT files?

- **PFM**: PFM, short for Portable Float Map, is an image file format that utilizes floating-point integers to store pixel values. It is commonly employed for high-dynamic-range photographs, storing pixel values with a wide range of intensities. PFM files can retain the entire spectrum of brightness values. In applications such as 3D reconstruction, depth mapping, object tracking, augmented reality, etc., PFM files are often used to store depth information for preserving intricate details and accurate depth representation.

- **MAT**: MAT files are a special type of file utilized by MATLAB, a popular tool renowned for its computational capabilities. These files hold information stored in a specific format, designed to store various types of data such as numbers, lists, and other forms of data efficiently.

## Reading PFM and MAT files

Our dataset comes from a depth camera and is formatted in PFM and MAT. To read these files, we utilize specific functions:

- **PFM files**: We use the "read_pfm" function to open the data from a PFM file and convert it into a NumPy array. This function first analyzes the file header to gather information about the format, dimensions, and scale factor. Then, the file content is read, verified, and decoded into a NumPy array. The orientation and scale of the array are adjusted before returning it.

- **MAT files**: To open a MAT file, we use the `scipy.io.loadmat` function, which loads the MATLAB files and extracts the label maps.

## 4.2. Encoder Steps

The encoder is a pivotal part of our project. It initiates by reading a PFM file (an image collected from a depth camera) as input. The encoder then performs a series of techniques:

1. Human extraction
2. Clothes extraction
3. Lossless coding
4. Residual image computation
5. Fixed rate layer determination
6. Enhancement layer creation
7. Bitstream append

These steps are illustrated in the sequential diagram provided in Figure 20.

The encoder's actions are mirrored by a decoder, which performs the actions in reverse order. It starts with the encoder's output, then decompresses and performs operations to reassemble the final image.

The entire coding and decoding workflow are visually represented in Figure 20, providing a comprehensive overview of the sequential steps involved.

Our aim is to evaluate the efficiency and effectiveness of this encoder-decoder pair on diverse hardware configurations, especially on Raspberry Pi platforms.

By evaluating the performance of the coder and decoder on a variety of hardware setups, we aim to identify the most efficient configuration for accomplishing the required compression and decompression tasks. This evaluation will provide valuable insights into the computational demands and resource utilization of the encoding and decoding processes, guiding optimization efforts for future implementations.

## Subject Extraction

After reading the PFM file, the next step is the extraction of the human subject from the image. The steps involved are depicted in Figure 21.

Extraction is a process of decomposing an image into its constituent parts. For complex images without a clearly defined application domain and content, extraction is particularly challenging. The precision of this step significantly impacts the success of more complex computer vision algorithms utilized in various analysis techniques.

The extraction code (shown in Fig. 21) starts by importing the necessary libraries (`cv2`, `numpy`, `matplotlib.pyplot`) and a custom `read_pfm` function that handles the reading of PFM files.

As an example, let's consider an image (see Figure 18) that depicts a human subject in a room with some furniture objects in the background. The goal is to extract the subject of the image. This is achieved by creating a binary mask using `cv2.inRange` function to isolate the human subject. The mask is then applied to the input image using bitwise operations, and the mean pixel value for the human segment is calculated. 

After the extraction, we'll have two segments: the background, coded with 0, and the human figure with its mean pixel value. The result can be seen in Figure 22.

Bit operations are widely employed in image processing to isolate or place an object on the background using a specified mask. On the mask image, the pixels representing our target object are labeled with 1, and those constituting the background are labeled with 0.

## Clothes Extraction

The purpose of this step is to take the image with the isolated human and extract the clothing articles present in it. Each clothing article's pixels will have the same value and color, differentiating them from other items. We created a color map for better visualization, assigning different colors to each clothing item for easy distinction.

We use `numpy.unique` to retrieve the unique labels and their frequencies from the label map for better understanding. Frequencies are stored in a dictionary with labels as keys and frequencies as values. After performing clothes extraction, we create a function for better visualization of the segmented clothes. Colors are assigned to labels in the code. The background color is set to white, and for the clothes, a color map is created to assign a unique color to each label.

The code generates an empty output image, `segm_clothes`, the same shape as the label map. It checks if each label occurs in the color dictionary. If it does, a mask is generated by comparing the label map to the current label, and the matching color is assigned to the `segm_clothes` pixels. The function concludes with the plotting of the new image, as seen in Figure 23.

```python
def clothes_segmentation(original_image_1, label_map):
    # Get the unique segments from the label map
    segments = np.unique(label_map)
    print(f"Unique segments: {segments}")
    
    # Define a new image for the segmented human with the mean pixel values
    segmented_clothes = np.zeros_like(original_image_1)
    
    # Loop over each segment
    for segment in segments:
        if segment == 0:
            continue  # Ignore the background
        mask = (label_map == segment)  # Create a mask for the current segment
        masked_img = original_image_1 * mask  # Apply the mask to the normal image
        segment_pixels = masked_img[mask]         
        mean_pixel_value = np.mean(segment_pixels)  # Calculate the mean pixel value in the masked image
        print(f"Mean pixel value of the segment {segment}: {mean_pixel_value}")
        # Apply the mean pixel value to the corresponding segment in the new image
        segmented_clothes[mask] = mean_pixel_value
    return segmented_clothes.astype(np.uint8)

segm_clothes = clothes_segmentation(original_image_1, label_map)
```

## Residual Image Calculation

The next step in the encoding process is calculating a residual image, which represents the difference between the segmented image and the segmented clothes image. This is achieved by subtracting the pixel mean of each similar clothing article from both images. Specifically, we apply a mask on the segmented image to isolate each clothing item, calculate the mean value of the pixels in that item, and subtract this value from the corresponding value in the clothes extraction image.

The function `apply_mask` is used for this purpose. This function takes the human subtraction image and the label map from the clothing extraction as inputs. It retrieves all unique labels from the label map and stores them in a dictionary. The function then iterates over each unique label, applies a mask to isolate the clothing item, and calculates the pixel mean value.

```python
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
```
To calculate the residual image, we substitute the values for the corresponding labels from both pictures.

At the end of this process, we print all the images from previous steps for clear visual representation and comparison of the results.

To further analyze the results, we recreate the video sequence from the dataset frames. First, we load the frames and normalize them to ensure that all pixel values are in the range of 0 to 1. We then convert the pixel values to 8-bit unsigned integers for accurate on-screen rendering. A loop is initiated to cycle through the three frames continuously, creating a seamless video sequence. The frames are displayed in a window titled 'Reconstructed video sequence', with a 500-millisecond delay between each frame. Pressing 'q' allows the user to stop the loop. The program concludes once the sequence ends or is interrupted by closing the display window.

## Arithmetic Encoder

Arithmetic coding is a lossless data compression method that encodes data using fractions in the [0, 1] range rather than binary codes, leading to optimized data representation.

In our workflow, the image is first normalized and scaled so that pixel intensities span from 0 to 255. We calculate the pixel intensity histogram and the cumulative distribution function, both essential for the encoding process.

Two functions, get_binaryStr_within_range and arith_encoding, are used in this process. The former returns a binary string for a range, while the latter applies the arithmetic coding algorithm to the flattened grayscale image data, generating binary 'tags' for each 8-pixel block. These tags are then converted into byte format, representing our compressed image.

This compression process significantly reduces data volume while preserving critical information, yielding a compression ratio of approximately 4. The necessary values are saved for use in the decompression process.

## Quantization 

Quantization is a process for reducing the number of distinct values in a signal to save memory and computational resources. In our workflow, we apply two quantization functions, quantizor_fixed_rate and quantizor_embedded, to an image, which allows for L∞ scalability and control over the error during compression and decompression.

```python
def quantizor_fixed_rate(x, delta_fixed_rate, epsilon):
    return np.where(np.abs(x) / delta_fixed_rate + epsilon > 0, np.round(np.sign(x) * (np.abs(x) / delta_fixed_rate + epsilon)), 0).astype(int)

def quantizor_embedded(x, delta_fixed_rate, epsilon, n):
    delta = delta_fixed_rate / 3
    return np.where(np.abs(x) / (3 ** n * delta) + epsilon / (3 ** n) > 0, np.round(np.sign(x) * (np.abs(x) / (3 ** n * delta) + epsilon / (3 ** n))), 0).astype(int)
```
We start with a maximum absolute difference (MAXAD) of 51 and decrease it by factors of 3 until reaching 1. The image then undergoes arithmetic coding, transitioning us to "near-lossless" compression.

After each iteration, we calculate the original image size and the compressed size, helping us illustrate the compression efficiency. With both quantization methods, the image size is reduced by approximately 98.43%, demonstrating the effectiveness of the process.

## Bitstream Append

In the final encoding step, we merge the compressed segmented clothes data and the quantization results into a singular bitstream. This simplifies the decoding process, allowing for it to be conducted in any programming language by simply reading the bitstream.

```python
bitstream = []
for n in range(MAXAD, 0, -1):
    bitstream.append(compress_with_fixed_rate(Dn))
    bitstream.append(compress_with_embedded_extension(Dn3n))
```

The bitstream starts with the compressed 'segm_clothes' data, followed by an iterative loop that appends two layers per cycle: a fixed-rate layer at delta 'Dn' and an embedded extension at 'Dn3n'. With each iteration, 'maxad' is decreased and 'n' is increased, resulting in an accurate representation of the sequence outlined in Figure 17.

Appending the bitstream in this manner ensures efficient encoding and facilitates a smooth decoding process later on.

## Decoder

An image decoder that reconstructs the original data from the encoder output.

The decoder reverses the encoding process, loading data such as image dimensions and compressed sizes to adapt to changes in the dataset or parameters. The decoder can be broken into the following steps:

### Decompressing Quantization Output
1. Convert binary strings back into decimal values.
2. Use these values with a cumulative distribution function to estimate initial pixel intensities.
3. Reshape the output to match the original image.

### Dequantization
1. Use the function described in Eq.21 to dequantize the data.
2. Understand that dequantized data will approximate the original, resulting in some quality loss due to data loss during quantization.

### Reconstructing the Residual Image
- Create an empty image.
- Loop through each label in the segmented image.
- Calculate and assign residual values to pixels, reconstructing the image.

### Decompressing "segm_clothes" Image
- Decode the encoded tags into a bit stream representing compressed image data.
-  Reshape the decoded stream to match the original image.

### Reconstructing the "segm_clothes" Image
- Identify unique segments in the image.
-  Calculate the mean pixel value for each segment.
-   Apply this mean value to the corresponding region in the reconstructed image.

### Reconstructing the "segm_img" Picture
- Initialize a new image.
- Extract unique labels from the label map and create a mask for each label.
- Add together the pixel values from the residual image and the reconstructed segmented clothes image for each segment in the new image.

### Reconstructing the Original Image
- Copy the original input picture.
- Retrieve the labels from the label map.
- Replace the segmented parts in the reconstructed original image with their counterparts from the reconstructed segmented image.
- Observe that the reconstruction will show some losses due to quantization, which can be observed in the final plots.

