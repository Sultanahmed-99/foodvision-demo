### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict


# Setup class names 
class_names = ['pizza' , 'steak' , 'sushi']

## Model and transforms prepreation ## 

# Create EffNetB2 model 

effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=3 # len(class_names) would also work !  
)

# Load Saved weights 
effnetb2.load_state_dict(
    torch.load(f = "09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth" , 
    map_location=torch.device("cpu")
    )

)

# Predict function # 

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """

    # start the timer 
    start_time = timer()

    # Transform the target image and add a batch dimension
    img  = effnetb2_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

  
# Crete the Gradio demo 


# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."


demo = gr.Interface(fn = predict , # mapping function from input to output 
                    inputs = gr.Image(type = "pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3 , label = "Predictions") ,
                             gr.Number(label="Prediction time (s)")] , # what are the outputs?
                    examples = exaple_list , 
                    title = title, 
                    description = description ,
                    )


# Launch the demo!
demo.launch()
