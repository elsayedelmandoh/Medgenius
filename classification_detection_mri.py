# Import necessary libraries
# For load classification model
import cv2
import numpy as np
import joblib

# For load detection model
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# For create two functions for utils 
import base64 # For function backgroumd 
import plotly.graph_objects as go # For function visualize

# For build GUI 
import streamlit as st 
from PIL import Image


# Function to perform "classification" using the first model
def predict_classification(image):
    # Load the classification model
    classification_model = joblib.load('model_classification_mri.h5')

    # Preprocess the image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (28, 28))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 28, 28, 1))

    # Use the loaded model for prediction
    probabilities = classification_model.predict(reshaped_image)
    predicted_label = np.argmax(probabilities)

    # Convert the predicted label to the corresponding class name
    result= 'Negative' if predicted_label == 0 else 'Positive'

    return result, probabilities


# Function to perform 'detection' using the second model
def perform_detection(image):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    # 
    cfg.MODEL.WEIGHTS = 'model_detection_mri.pth'
    cfg.MODEL.DEVICE = 'cpu'

    predictor = DefaultPredictor(cfg)

    image_array = np.asarray(image)

    # Detect objects using the second model
    outputs = predictor(image_array)

    threshold = 0.6

    preds = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    bboxes = outputs["instances"].pred_boxes

    bboxes_ = []
    for j, bbox in enumerate(bboxes):
        bbox = bbox.tolist()
        score = scores[j]
        pred = preds[j]

        if score > threshold:
            x1, y1, x2, y2 = [int(i) for i in bbox]
            bboxes_.append([x1, y1, x2, y2])

    return bboxes_


# First function to set background image for Streamlit app
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# Second function to visualize the detections
def visualize(image, bboxes):
   
    # Get the width and height of the image
    width, height = image.size

    shapes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # Convert bounding box coordinates to the format expected by Plotly
        shapes.append(dict(
            type="rect",
            x0=x1,
            y0=height - y2,
            x1=x2,
            y1=height - y1,
            line=dict(color='red', width=6),
        ))

    fig = go.Figure()

    # Add the image as a layout image
    fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch"
        )]
    )

    # Set the axis ranges and disable axis labels
    fig.update_xaxes(range=[0, width], showticklabels=False)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        range=[0, width], showticklabels=False
    )

    fig.update_layout(
        height=800,
        updatemenus=[
            dict(
                direction='left',
                pad=dict(r=10, t=10),
                showactive=True,
                x=0.11,  #0.11 x width of the figure 
                xanchor="left",
                y=1.1,
                yanchor="top",
                type="buttons",
                buttons=[
                    dict(
                        label="Original",
                         method="relayout",
                         args=["shapes", []], # hide shape 
                    ),  
                    dict(
                        label="Detections",
                         method="relayout", # is used to update the layout of an existing figure
                         args=["shapes", shapes], # display shape or annotation bounding box
                    ),
                ],
            )
        ]
    )

    st.plotly_chart(fig)


# Build streamlit app
# set_background(r'data\bg.png')
st.title('Brain MRI Tumor Detection and Classification')
st.header('Please upload an image')

file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

if file:
    # Load uploaded image
    image = Image.open(file).convert('RGB')

    # Perform classification using the first model
    result, probabilities = predict_classification(np.array(image))
    print(f"Predicted Classification: '{result}'")
    print(f"probabilities: '{probabilities}'")
    
    # Check if the first model predicted 'Positive'
    if result == 'Positive':
        # Perform detection using the second model
        bboxes_ = perform_detection(np.array(image))

        # Visualize the detections
        st.markdown("<span style='color:red;'><strong>Opps!</strong></span>\n\n<span style='color:red;'><strong>We detect tumor!</strong></span>\n\n<span style='color:green;'><strong>Press on button 'Detections' to display tumor.</strong></span>", unsafe_allow_html=True)
        visualize(image, bboxes_)
        
    else: 
        # If the first model predicted 'Negative', and display the uploaded image
        st.image(image)
        st.markdown("<span style='color:green;'><strong>Great!</strong></span>\n\n<span style='color:red;'><strong>No tumor detected.</strong></span>\n\n<span style='color:red;'><strong>Your test is good.</strong></span>", unsafe_allow_html=True)
