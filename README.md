# Chest X-Ray Abnormality Detection 
## Multi Label Convolutional Neural Network
## KENDALL MCNEIL
November 2023

![xrays](https://github.com/kmcneil901/Chest-X-Ray-Abnormality-Detection-Multi-Label-CNN/assets/139075900/8f09dfde-d263-4b91-88e0-92069c54035d)

## Overarching Project Description
OBJECTIVE & METHODOLOGY: The chest x-ray is one of the most challenging to interpret, which can result in misdiagnosis even for seasoned healthcare providers. Building a strong convolutional neural network (CNN) to detect common thoracic lung diseases in chest x-rays would improve diagnostic accuracy for patients and ultimately save lives through early and accurate detection. The CNN will act as an automated system to support radiologists as a second opinion in reviewing chest x-rays for abnormalities. The work product will alleviate the stress of busy doctors and healthcare providers while also providing patients with a more accurate and efficient diagnosis. The objective, therefore, is to detect a variety (14 total) of common thoracic lung abnormalities in chest x-rays by building an AI system using a Convolutional Neural Network (CNN). The multi-label neural network image classification model was designed using Tensorflow. 

AUDIENCES: The general target audience for the project is the healthcare industry. The more specific presentation audience is Vingroup Big Data Institute (VinBigData) that is working to build large-scale and high-precision medical imaging solutions based on the latest advancements in AI to facilitate efficient clinical workflows.

DATA: VinBigData has provided a dataset of 18,000 CXR scans dicom images labeled by a panel of experienced radiologists for the presence of 14 common thoracic abnormalities: aortic enlargement, atelectasis, calcification, cardiomegaly, consolidation, ILD, infiltration, lung opacity, nodule/mass, other lesion, pleural effusion, pleural thickening, pneumothorax, and pulmonary fibrosis. The dataset was created by assembling de-identified chest X-ray studies provided by two hospitals in Vietnam: the Hospital 108 and the Hanoi Medical University Hospital.

## Repository Outline
1. "Chest X-Ray Abnormality Detection (Multi-Label CNN).ipynb" is the main notebook page where the data cleaning, preprocessing, and modeling takes place.
2. The PDF titled "Lung Abnormality Detection in Chest X-Rays Presentation Deck" are the slides from the non-techical presentation. 
3. "lung_abnormality_detection_app.py" is where the steamlit website creation code is located.
4. "model.h5" is the final model that is linked to the deployed streamlit website to generate new predictions.
5. "requirements.txt" is the documentation about the environment setup to ensure seamless deployment of the streamlit webste despite any environment dependencies.
6. The "app_images" folder is where all the images used on the streamlit website are located.

## Model Results
After preprocessing the data and testing our over 30 models using Conv2D, MaxPooling, Dropout and other regularization techniques, and image augmentation, the final results of the "black box" model was an 84% accuracy rate on the testing set and 85% accuracy rate on the training set. This is a strong start. Next steps include continued model tuning and pulling more data into the model. 

## Final Deliverable
For this project, I really desired to create something tangible that any user could have access to. Therefore, I leveraged steamlit to create [The Lung Abnormality Detection website](https://chest-x-ray-abnormality-detection-multi-label-cnn-hlxsplvft7v3.streamlit.app/) where a user submits a chest x-ray and then the model generates predictions and displays the results to the user. Additionally, for a user interested in the website, but does not have a chest x-ray, I provided four example chest x-rays. 

<img width="733" alt="website_pic2" src="https://github.com/kmcneil901/Chest-X-Ray-Abnormality-Detection-Multi-Label-CNN/assets/139075900/2922b573-8396-4782-8be7-008909f0161f">

##Thank you! 
Thank you for taking the time to explore my project. Be sure to checkout the website linked above and connect on [LinkedIn](https://www.linkedin.com/in/kendallmcneil/). If you have any feedback, please contact me using the information provided on the website. Cheers! 
