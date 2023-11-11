import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os
import streamlit as st
import tensorflow as tf 

# Get the current working directory
current_dir = os.getcwd()

# Specify the path to the model file relative to the working directory
model_path = os.path.join(current_dir, 'final_df.pkl')

# Print information for debugging
st.write(f"Current working directory: {current_dir}")
st.write(f"Model file path: {model_path}")

# Check if the file exists
if os.path.exists(model_path):
    # Load the trained model
    model_file = open(model_path, 'rb')
    loaded_model = joblib.load(model_file)
    model_file.close()
    st.success("Model loaded successfully.")
else:
    st.error(f"Model file not found at path: {model_path}")


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
    9: 'Abnormality Detected: Other lesion',
    10: 'Abnormality Detected: Pleural effusion',
    11: 'Abnormality Detected: Pleural thickening',
    12: 'Abnormality Detected: Pneumothorax',
    13: 'Abnormality Detected: Pulmonary fibrosis',
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
def make_prediction(model, input_data, threshold=0.6):
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

    st.write("The chest radiograph is one of the most challenging to interpret, resulting in misdiagnosis even for the most seasoned of healthcare providers. By leveraging artificial intelligence, this automated tool aims to enhance diagnostic accuracy, alleviate stress of healthcare providers, and, ultimately, improve patient outcomes through early and accurate detection of lung abnormalities.")

    # Image at the top
    st.image('xrays.png', use_column_width=True)

    # Describe the objective
    st.write('Submit a chest x-ray to detect up to 14 of the most common lung abnormalities.')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        input_data = preprocess_image(image)

        # Make predictions
        predictions = make_prediction(loaded_model, input_data)

        # Check if predictions is not None
        if predictions is not None:
            # Display the predictions
            st.write("### Results:")
            abnormalities_detected = False
            for idx, (abnormality, probability) in enumerate(zip(class_mapping.values(), predictions[0])):
                if probability > 0.5 and idx < 14:  # Exclude class 14
                    st.write(f"{abnormality}")
                    abnormalities_detected = True

            if not abnormalities_detected:
                st.write("No Abnormalities Detected")

            # Add a line space
            st.write("")

            # Add the subheader with a smaller font
            st.write("*Important: Be sure to consult a doctor regarding these results.*", font="italic small")

# Sidebar - "About the Creator"
st.sidebar.title('About the Creator')
st.sidebar.image("headshot.jpg", use_column_width=True)

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
