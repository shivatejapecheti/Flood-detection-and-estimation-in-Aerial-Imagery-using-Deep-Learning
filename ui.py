import cv2
import numpy as np
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import load_model

N = 512
# Function to generate masked image
def generate_masked_image(input_image, model):
    # Convert the image to numpy array
    img = np.array(input_image)
    
    # Resize the image
    img = resize(img, (N, N,3), mode="constant", preserve_range=True)
    
    # Normalize pixel values
    # img = img / 255.0
    
    # Expand dimensions to match the model input shape
    img = np.expand_dims(img, axis=0)
    
    # Generate the masked image
    pred_mask = model.predict(img, verbose=0)
    masked_image = np.squeeze(pred_mask)
    
    # Create a figure and plot the masked image
    fig = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(masked_image)
    plt.show()
    
    # Convert the figure to a PIL Image
    fig.canvas.draw()
    masked_image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    
    return masked_image

# Function to preprocess the image
def preprocess_image(img):
    # Resize image
    img = cv2.resize(img, (64, 64))
    
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    img = img.astype("float32") / 255.0
    
    # Add an extra dimension to match the model's input shape
    img = np.expand_dims(img, axis=0)
    
    return img

# Load the model from an H5 file

# Streamlit app
st.title("Image Prediction")
st.write("Upload an image and let the model make predictions.")

def calculate_flooded_percentage(image):
    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # Extract the green channel (assuming the flooded area is represented by green)
    green_channel = img_array[:, :, 1]

    # Binarize the green channel to have 0 for unflooded area and 255 for flooded area
    _, binary_mask = cv2.threshold(green_channel, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the total number of pixels in the mask
    total_pixels = np.prod(green_channel.shape)

    # Count the number of white pixels (flooded area) in the binary mask
    flooded_pixels = cv2.countNonZero(binary_mask)

    # Calculate the percentage of the flooded area
    percentage_flooded = (flooded_pixels / total_pixels) * 100

    return percentage_flooded

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_model('mlp')
    # Read the image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_image)
    print(predictions)
    # Display the image
    st.image(image, channels="RGB", caption="Uploaded Image", use_column_width=True)
    
    # Display the predictions
    if predictions <= 0.5:
        model_path = "unetModel"
        model = tf.keras.models.load_model(model_path)
        # uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        # Load the image
        input_image = Image.open(uploaded_file)

        # Generate the masked image
        masked_image = generate_masked_image(input_image, model)

        # Display the original and masked images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(input_image, use_column_width=True)
        with col2:
            st.subheader("Masked Image")
            st.image(masked_image, use_column_width=True)
            out = np.array(masked_image)
            out1 = out.flatten()
            # st.write(out1.shape,out.shape)
            st.write(calculate_flooded_percentage(masked_image))
        
    else:
        st.write("No flood found")


