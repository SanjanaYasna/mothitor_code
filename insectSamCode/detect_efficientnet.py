from detection import save_detections, detect 
from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import requests
import torch
import torchvision.models as models
import torch.nn as nn 
from torchvision import transforms
import pandas as pd
# cuda flags ,if you get OOM:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

csv_dir ="/work/pi_mrobson_smith_edu/mothitor/code_main/efficientnet_data/results/limited_data_more_moths_run/detections.csv"
labels = ["insect"]
threshold = 0.2

detector_id = "IDEA-Research/grounding-dino-base"
segmenter_id = "martintmv/InsectSAM"

#efficientnet model loaded (total dataset model) 
transform_tensor = transforms.Compose([
    transforms.ToTensor()])
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

image_dir = "/work/pi_mrobson_smith_edu/mothitor/data/Mothitor4.0Pics" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#detections.csv contains the info: image_name, detection_score, lable, xmin, ymin, xmax, ymax 
#if csv has no lines, then add header
if not os.path.exists(csv_dir):
    with open(csv_dir, "w") as f:
        f.write("image_name,detection_score,label,xmin,ymin,xmax,ymax\n")
        f.flush()
    f.close()

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def pass_at_5(cropped_image, cutoff = 0.999): 
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
def predict_with_cutoff(cropped_image, cutoff = 0.9999):
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(cropped_image))
    output = output.cpu().detach().numpy()
    #if first value of output is greater than cutoff, then it is a moth, otherwise not a moth 
    if output[0][0] > cutoff:
        pred_class_name = "moth"
    else:
        pred_class_name = "not_moth"
    return pred_class_name
            
    
def detection_with_efficientnet(
    image: Union[Image.Image, str],
    img_name: str,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None,
):
    if isinstance(image, str):
        image = load_image(image)
    detections = detect(image, labels, threshold, detector_id) 
    #ONE BY ONE TO AVOID RESHAPE TRANSFORM (presuambly more accurate))
    for detection in detections:
        #get score, xmin, ymin, xmax, ymax 
        box = detection.box
        score = detection.score
        xmin = box.xmin 
        ymin = box.ymin
        xmax = box.xmax
        ymax = box.ymax
        #crop out the image for that specific detection
        #crop coordinates described left, upper, right, and lower pixel
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_image_save = cropped_image.copy()
        #transform to tensor, make 4D (since not batched...)
        cropped_image = transform_tensor(cropped_image).unsqueeze(0)
        cropped_image = cropped_image.to(device)
        pred_class_name, max_score = pass_at_5(cropped_image)  
        with open(csv_dir, "a") as f:
            f.write(f"{img_name},{score},{pred_class_name},{xmin},{ymin},{xmax},{ymax}\n")
            f.flush()
        f.close()
        
#singe image try
df = pd.read_csv(csv_dir)
images_already_done = df["image_name"].tolist()
image_dir = "/work/pi_mrobson_smith_edu/mothitor/data/Mothitor4.0Pics"
for image_name in os.listdir(image_dir):
    if image_name in images_already_done:
        continue
    detection_with_efficientnet(
        image=image_dir + "/" + image_name,
        img_name = image_name,
        labels=labels,
        threshold=threshold,
        detector_id=detector_id
    )
