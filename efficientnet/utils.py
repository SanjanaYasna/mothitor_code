from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import requests
import torch
import torchvision.models as models
import torch.nn as nn 
from torchvision import transforms
import pandas as pd


def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image


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

def pass_at_5(model, cropped_image, cutoff = 0.999): 
    predictions = []
    for i in range(5):
        with torch.no_grad():
            output = torch.nn.functional.softmax(model(cropped_image))
        output = output.cpu().detach().numpy()
        predictions.append(output[0][0])
    #if all predictions greater than 0.99, then it is a moth
    if all(pred > cutoff for pred in predictions):
        pred_class_name = "moth"
    else:
        pred_class_name = "not_moth"
    max_score = max(predictions)
    return pred_class_name, max_score


#set prediction cutoff
def predict_with_cutoff(model, cropped_image, cutoff = 0.5):
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(cropped_image))
    output = output.cpu().detach().numpy()
    #if first value of output is greater than cutoff, then it is a moth, otherwise not a moth 
    if output[0][0] > cutoff:
        pred_class_name = "moth"
    else:
        pred_class_name = "not_moth"
    return pred_class_name