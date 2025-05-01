from segmentation import grounded_segmentation
from detection import save_detections
from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
import os
# cuda flags ,if you get OOM:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    
    
#image array dimensions: (6944, 9152, 3)
image_url = "/work/pi_mrobson_smith_edu/mothitor/main/mothitor_yolo/ama_2024-06-17_23_00_04.jpg"
labels = ["insect"]
threshold = 0.2

detector_id = "IDEA-Research/grounding-dino-base"
segmenter_id = "martintmv/InsectSAM"

image_array, detections = grounded_segmentation(
    image=image_url,
    labels=labels,
    threshold=threshold,
    polygon_refinement=True,
    detector_id=detector_id,
    segmenter_id=segmenter_id
)


#save image
# image_array = Image.fromarray(image_array)
save_detections(
    image=image_array,
    detections=detections,
    save_path="/work/pi_mrobson_smith_edu/mothitor/main/mothitor_yolo/insectsam_moth_two_white_moths.jpeg"
)
