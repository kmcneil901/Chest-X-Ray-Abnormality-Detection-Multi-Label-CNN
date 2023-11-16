import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import time

# Specify the path to the model file relative to the working directory
model_path = 'model.h5'
loaded_model = tf.keras.models.load_model(model_path)

# Mapping from class indices to abnormality names
class_mapping = {
    0: 'Abnormality Detected: Aortic Enlargement',
    1: 'Abnormality Detected: Atelectasis',
    2: 'Abnormality Detected: Calcification',
    3: 'Abnormality Detected: Cardiomegaly',
    4: 'Abnormality Detected: Consolidation',
    5: 'Abnormality Detected: Interstitial Lung Disease (ILD)',
    6: 'Abnormality Detected: Infiltration',
    7: 'Abnormality Detected: Lung Opacity',
    8: 'Abnormality Detected: Nodule/Mass',
    9: 'Abnormality Detected: Other Lesion',
    10: 'Abnormality Detected: Pleural Effusion',
    11: 'Abnormality Detected: Pleural Thickening',
    12: 'Abnormality Detected: Pneumothorax',
    13: 'Abnormality Detected: Pulmonary Fibrosis',
    14: 'No Abnormalities Detected'
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Resize the image to the required input size of your model
        resized_image = image.resize((256, 256))
        # Convert the image to a NumPy array
        image_array = np.array(resized_image)
        # Normalize pixel values to be between 0 and 1
        normalized_image = image_array / 255.0
        # Expand dimensions to create a batch size of 1
        input_data = np.expand_dims(normalized_image, axis=0)
        # Ensure the input shape matches the model's expectations
        input_data = np.expand_dims(input_data, axis=-1)
        return input_data
    except Exception as e:
        st.warning("Please upload a valid radiograph.")
        return None

# Function to make predictions using the model
def make_prediction(model, input_data, threshold=0.5):
    try:
        # Make predictions
        predictions = model.predict(input_data)
        # Apply threshold for binary classification
        binary_predictions = (predictions > threshold).astype(int)
        return binary_predictions
    except ValueError as e:
        st.warning("Please upload a valid radiograph.")
        return None

# Main Streamlit app
def main():
    st.title("Lung Abnormality Detection")

    st.write("The chest x-ray is one of the most challenging to interpret even for the most seasoned of healthcare providers with a false negative rate of 20-30%. By leveraging artificial intelligence, this automated tool aims to alleviate stress of healthcare providers and ultimately improve patient outcomes through early and accurate detection of lung abnormalities.")

    # Insert a line or divider
    st.markdown("---")
    
    # Test out some of our images
    st.subheader("Test It Out")
    
    # Load your chest X-ray images (replace these paths with your actual file paths)
    image_paths = ["app_images/image1.png", "app_images/image3.png", "app_images/image2.png", "app_images/image4.png"]

    # Function to resize the image to a specified width and height
    def resize_image(image_path, width, height):
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        return resized_image
    
    # Specify the width and height for resizing
    image_width = 400
    image_height = 400
    
    # Use st.columns to create columns
    columns = st.columns(len(image_paths))
    
    # Mapping of image indices to results
    results_mapping = {
        1: "Abnormalities Detected: Lung Opacity, Nodule/Mass, Infiltration, Consolidation",
        2: "Abnormality Detected: Interstitial Lung Disease (ILD)",
        3: "All Clear - No Abnormalities Detected",
        4: "Abnormalities Detected: Cardiomegaly, Aortic Enlargement, Pleural Effusion, Pleural Thickening, Other Lesion"
    }
    
    # Display each image in a column with a "Detect" button underneath
    for idx, (column, image_path) in enumerate(zip(columns, image_paths)):
        # Resize the image
        resized_image = resize_image(image_path, image_width, image_height)
        
        # Display the resized image with the specified width and center it
        column.image(resized_image, caption=f"Chest X-Ray {idx + 1}", use_column_width=True)
        
        # Add a "Detect" button centered below the image
        if column.button(f"Detect {idx + 1}"):
            # Display the spinner while processing
            with st.spinner(f"Results are loading for chest x-ray {idx + 1}..."):
                # Simulate model processing time (replace with your actual detection logic)
                time.sleep(3)
        
                # Your detection logic goes here
                result = results_mapping.get(idx + 1, "Unknown")  # Replace with your actual result
        
                # Display the result
                column.write(f"{result}")
    
    # Insert spacing with divider 
    st.markdown("---")
    
    st.subheader('Submit Your Own')
    st.write('Submit your own chest x-ray to detect up to 14 of the most common lung abnormalities.')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.")
    
        # Preprocess the image
        input_data = preprocess_image(image)
    
        # Add a "Detect" button
        if st.button("Detect Abnormalities"):
            
            # Display the spinner while processing
            with st.spinner("Model is making predictions..."):
                # Simulate model processing time (replace with your actual detection logic)
                time.sleep(3)
                
                # Make predictions
                predictions = make_prediction(loaded_model, input_data)
    
                # Check if predictions is not None
                if predictions is not None:
                    # Display the predictions
                    st.write("### Results:")
                    abnormalities_detected = False
                    for idx, (abnormality, probability) in enumerate(zip(class_mapping.values(), predictions[0])):
                        if probability > .5 and idx < 14:  # Exclude class 14
                            st.write(f"{abnormality}")
                            abnormalities_detected = True
    
                    if not abnormalities_detected:
                        st.write("No Abnormalities Detected")
    
                    # Add a line space
                    st.write("")
    
                    # Add the subheader with a smaller font
                    st.write("*Important: Be sure to consult a doctor regarding these results.*", font="italic small")

    st.write("   ")
    st.markdown("---")

    github_project_url = "https://github.com/kmcneil901/Chest-X-Ray-Abnormality-Detection-Multi-Label-CNN"
    github_project_markdown = f'[GitHub]({github_project_url})'

    st.write("   ")
    st.write(f"This A.I. tool is based on a convolutional neural network (CNN) image classification model using Python, Tensorflow, and Keras. The model currently has an 85% accuracy rate and can be found in {github_project_markdown}. Please feel free to connect with me on LinkedIn or via email. Feedback is welcome!") 

# Sidebar - "About the Creator"
st.sidebar.title('About the Creator')
st.sidebar.image("app_images/headshot.jpg", use_column_width=True)

linkedin_url = "https://www.linkedin.com/in/kendallmcneil/"
github_url = "https://github.com/kmcneil901?tab=stars"
medium_url = "https://medium.com/@kendallmmcneil"

linkedin_markdown = f'[LinkedIn]({linkedin_url})'
github_markdown = f'[GitHub]({github_url})'
medium_markdown = f'[Blog]({medium_url})'

st.sidebar.subheader('Kendall McNeil')
st.sidebar.markdown(f"{linkedin_markdown} | {github_markdown} | {medium_markdown}", unsafe_allow_html=True)
st.sidebar.write('kendallmmcneil@gmail.com')

# Run the app
if __name__ == "__main__":
    main()
