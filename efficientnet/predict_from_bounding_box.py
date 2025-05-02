import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image
import time
import requests 
import os
import pandas as pd
from torchvision import datasets, transforms
#below for batch_size > 1
transform = transforms.Compose([
    transforms.Resize((90, 110)),
    transforms.ToTensor(),
])
batch_size = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights='DEFAULT')
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
model.load_state_dict(torch.load('/work/pi_mrobson_smith_edu/mothitor/binary_dataset/efficientnet/total/model_tensors/epoch_49_tensors',
                                 map_location=device,
                                  weights_only=True)
                     ) 
model = model.to(device)   
model.eval() 


#class map: {'moth': 0, 'non_moth': 1}
class_names = ["moth", "not_moth"] 


#set csv 
csv_dir = "/work/pi_mrobson_smith_edu/mothitor/code_main/efficientnet_data/no_reshape_pass_5/detections.csv"
with open(csv_dir, "a") as f:
    f.write("image_name,detection_score,label,xmin,ymin,xmax,ymax\n")
    f.flush()
f.close()
    
image_dir = "/work/pi_mrobson_smith_edu/mothitor/data/Mothitor4.0Pics" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["moth", "not_moth"]

def pass_at_5_by_max(cropped_image): 
    predictions = []
    for i in range(5):
        with torch.no_grad():
            output = model(cropped_image)
        output = output.cpu().detach().numpy()
        #get index of max value
        pred = np.argmax(output, axis=1)
        predictions.append(pred[0])
    #choose class by 3 or more predictions
    #if 3 or more 0, then moth
    #if 3 or more 1, then not moth
    if predictions.count(0) >= 3:
        pred_class_name = "moth"
    else:
        pred_class_name = "not_moth"
    return pred_class_name
            
"predict from bounding box, calling pass @ 5"
def detect_from_bounding_box(
    img_name: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float
):
    image = Image.open(img_name).convert("RGB")
    #crop image
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    #transform to tensor WITH reshape, make 4D (since not batched...)
    cropped_image = transform(cropped_image).unsqueeze(0)
    cropped_image = cropped_image.to(device)
    pred_class_name = pass_at_5_by_max(cropped_image)
    output = model(cropped_image)
    output = output.cpu().detach().numpy() 
    return pred_class_name

#read data of bounding boxes from groundingDINO
df = pd.read_csv('/work/pi_mrobson_smith_edu/mothitor/code_main/efficientnet_data/results/mothitor_pass_5_cutoff_0.9999/detections.csv')


for index, row in df.iterrows():
    img_name = row['image_name']
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    detection_score = row['detection_score']  
    #image_name makes a path variable
    image_path = os.path.join(image_dir, img_name)
    pred_class = detect_from_bounding_box(image_path, xmin, ymin, xmax, ymax) 
    with open(csv_dir, "a") as f:
        f.write(f"{img_name},{detection_score},{pred_class},{xmin},{ymin},{xmax},{ymax}\n")
        f.flush()
    f.close()
    