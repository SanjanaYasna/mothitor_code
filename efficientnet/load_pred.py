import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
#below for batch_size > 1
transform = transforms.Compose([
    transforms.Resize((90, 110)),
    transforms.ToTensor(),
])
#below for oneby-one
transform_tensor = transforms.Compose([
    transforms.ToTensor()])
batch_size = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
model.load_state_dict(torch.load('/work/pi_mrobson_smith_edu/mothitor/binary_dataset/efficientnet/model_tensors/try/epoch_1_1_tensors',
                                 map_location=device,
                                  weights_only=True)
                     ) 
model = model.to(device)   
model.eval() 
test_dataset = datasets.ImageFolder('/work/pi_mrobson_smith_edu/mothitor/binary_dataset/test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class_names = ["moth", "not_moth"]
#CAN DO ONE BY ONE TO AVOID RESHAPE TRANSFORM (presuambly more accurate))
for batch_idx, (data, target) in enumerate(test_loader):
    data = data.to(device)
    output = model(data)
    print(output)
    #TODO; revise below, not very efficient 
    output = output.cpu().detach().numpy() 
    #get max for classes
    pred = np.argmax(output, axis=1)
    #get class names
    pred_class_name = [class_names[i] for i in pred]
    print(f"Predicted class: {pred_class_name}")
    break