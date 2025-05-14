import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read B&W image and load the caffemodel
frame = cv.imread("new.jpg")
if frame is None:
    print("Error loading image.")
    exit()

# Load the numpy file and caffe model
numpy_file = np.load('pts_in_hull.npy')
Caffe_net = cv.dnn.readNetFromCaffe("./models/colorization_deploy_v2.prototxt", 
                                    "./models/colorization_release_v2.caffemodel")

# Check if model loaded successfully
if Caffe_net.empty():
    print("Error loading the Caffe model.")
    exit()

# Add layers to the Caffe model using pts_in_hull.npy
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)  # Check this reshape, should be correct
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Extract the L channel from the image and preprocess
input_width = 224
input_height = 224
rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)

# Convert to Lab color space (L is the luminance channel)
lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
l_channel = lab_img[:,:,0]

# Resize L channel and normalize
l_channel_resize = cv.resize(l_channel, (input_width, input_height)) 
l_channel_resize -= 50  # Normalize L channel

# Prepare input blob
input_blob = cv.dnn.blobFromImage(l_channel_resize, scalefactor=1.0, size=(input_width, input_height), mean=(0, 0, 0), swapRB=False, crop=False)
Caffe_net.setInput(input_blob)

# Perform forward pass
try:
    output = Caffe_net.forward()
    print(f"Output Shape: {output.shape}")
    ab_channel = output[0, :, :, :].transpose((1, 2, 0))

    # Resize the AB channel to the original image size
    (original_height, original_width) = rgb_img.shape[:2]
    ab_channel_us = cv.resize(ab_channel, (original_width, original_height))

    # Concatenate L channel and AB channel to form LAB image
    lab_output = np.concatenate((l_channel[:,:,np.newaxis], ab_channel_us), axis=2)

    # Convert LAB to BGR for display
    bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)

    # Save the output image
    cv.imwrite("./result.png", (bgr_output*255).astype(np.uint8))
    print("Colorized image saved as result.png")

except Exception as e:
    print(f"Error during forward pass: {e}")
