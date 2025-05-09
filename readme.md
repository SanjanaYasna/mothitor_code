# Project File Organization:

REPO LINK: https://github.com/SanjanaYasna/mothitor_code.git (in case  writeup/paper copy.pdf doesn't show in gradescope assignment)

## Paper for project overview (recommended as a start)

Located in writeup/paper copy.pdf

## Presentation Slides: 

Located in presentation/is it a moth.pdf

# Code Organization for Important Files:

NOTE: all these files were done on a remote environment with abolute paths to folders for personal convenience. In order to run the model yourself, it is recommended that you adjust file paths to what matches your workspace.

## efficientnet folder:


```draw_bounding_box.py```: Code for taking detection results from groundingDINO and EfficientNet Full classifications (a fine-tuned model this project offers) in the form of a csv and drawing bounding boxes from the results, with labels of "moth", "not_moth", and detection score 

```predict_from_bounding_box.py```: From bounding box coordinates (From groundingDINO), return classification predictions (here, EfficientNet Full) 

**draw_bounding_box.py paired with predict_from_bounding_box.py make overall pipeline used, but insectSamCode/detect_efficientnet.py has consolidated form of this for use**

```train.py```: Fine-tuning script of the EfficientNet models on the AMI-GBIF binary dataset

```utils.py```: Prediction utilties. pass @ 5 by max score or by cutoff score, and pass @ 1 with a cutoff for moth class label

## efficientnet_data/results folder: 

```limited_data_more_moths``` folder has train.csv and test.csv that logs the metrics collected during the train and test process for the EfficientNet Limited model, trained on the limited dataset of AMI-BGIF binary (these descriptions are expanded upon in the paper) 

```total_data_run``` folder has train.csv and test.csv that logs for EfficientNet Full model, trained on the full AMI-BGIF binary dataset (these descriptions are expanded upon in the paper) 

```notebooks/ folder``` contains interactive notebooks used for parsing detection results, train and test metrics, or annotating images. Not scripts formally run,but for personal debugging or visualization purposes

```imgData.csv``` metadata of Macleish images

## AMI-GBIF Binary dataset metadata: 

``` total_test.csv``` contains the image names for shared test set among EfficientNet Full and Limited. init_test.csv is a duplicate, don't mind it... 

```overall_binary_labels.csv``` contains all image names and labels (moth or not).  1 is moth, 0 is non-moth, but these labels were flipped when organizing class folder splits to 0 for moth, 1 for non-moth, as is defined by EfficientNet

```init_train.csv``` contains image metadata for the Limited dataset subset used.  0 for moth, 1 to non-moth, as is defined with EfficientNet model predictions

### efficientnet_data/reshape_pass_5.csv:

These are the detections from EfficientNet Full used to answer questions regarding insect prescence in Macliesh images at different times the traps were activated

## insectsamCode folder:

Contains mix of code adapted from InsectSAM project (https://github.com/martintomov/RB-IBDM/tree/main), with additional code for integration with EficientNet models

```detect_efficientnet.py```:  Call GroundingDINO to get detections, and EfficientNet Full after to classify, for use if bounding boxes haven't been logged ahead of time, just like detect_efficientnet.py script. This is basically entire ml pipeline: a combination of draw_bounding_box.py and predict_from_bounding_box.py

```detection.py```: GroundingDINO code for image detection/annotation

```segmentation.py```: InsectSAM code for crop segmentation, but not used

The other files weren't used/run much, so not described.

# Model Availability: 

EfficientNet Full and EfficientNet limited models found in huggingface: https://huggingface.co/sanyasna517/mothitor_ml_models

GroundingDINO loaded from https://huggingface.co/IDEA-Research/grounding-dino-base 
