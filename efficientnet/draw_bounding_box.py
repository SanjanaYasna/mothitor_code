import pandas as pd
import numpy as np
import os
from PIL import Image
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union

import cv2
import torch
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import  pipeline 

df = pd.read_csv("/work/pi_mrobson_smith_edu/mothitor/code_main/efficientnet_data/results/mothitor_pass_5_cutoff_0.9999/detections.csv")
#read the first 100 unique image_name rows
first_100 = df['image_name'].unique()[:100]
# Create a new DataFrame with the first 100 unique image_name rows
df = df[df['image_name'].isin(first_100)]


#take the detection results -> return annoated image 
def annotate(image, annotations_df) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = Image.open(image).convert("RGB")
    image_cv2 = np.array(image_cv2)
   # image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
   # image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes from annoations_df
    for index, row in annotations_df.iterrows():
        label = row['label']
        score = row['detection_score']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (xmin, ymin), (xmax, ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color.tolist(), 3)
    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


annotation_dir = "/work/pi_mrobson_smith_edu/mothitor/code_main/efficientnet_data/results/mothitor_pass_5_cutoff_0.9999/visualize_100"

#go over images by all rwos pertaining to same image_name
for img in first_100:
    #get all rows pertaining to same image_name
    df_sub = df[df['image_name'] == img]
    #get image path
    image_path = os.path.join("/work/pi_mrobson_smith_edu/mothitor/data/Mothitor4.0Pics", img)
    #annotate image
    image = annotate(image_path, df_sub)
    #save image
    image = Image.fromarray(image)
    print("Saving image:", img)
    image.save(os.path.join(annotation_dir, img))
    