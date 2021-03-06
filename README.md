# COSI 101 FINAL PROJECT

## How we trained symbols
* Step 1: ./data_processing contains files to process images and extract symbols from 
  - run preprocess_images.py to shrink images to 28x28 and save to its current position
  - run extract_symbol.py to get train and test data

* Step 2: ./data_training contains files to train images
  - train_model.py to train train data and create a checkpoint for the CNN deep learning model
  - validate_test_data.py to validate the model on test data created

* notes: we deleted data in our submission so we don't exceed file limit
## Instruction: 

* Step 1: run clipImage.py
  - clipImage.py and clip_brighten.py are 2 files to modify original hand written images such that the output images need to be B&W with black screen and white mark.
  - clipImage takes a folder of image (line 8), and modify it to become black screen and white mark. This file is writen with the assumption that the images have white screen and black mark. In the code, we do a cv2.bitwise_not() to transform black and white pixel. If the input data has black background already, please comment out the line. 
  - Line 17-20 of this file is to make the image B&W. The threshold is currently set to 90, but it might be different for each image, so the clip_brighten.py file is to handle that issue.
  - The output folder of clipImage.py is ./result
  - clip_brighten.py take in an image file name (line 10) and apply the same modification. The only difference is that you can set the threshold for each image. So if you see an output image generated from clipImage.py is fully black or fully white, try adjusting the threshold (line 22) until the expression shows up clearly.

* Step 2: ./data_prediction contains files to predict images
  - predict.py is our "main" function to predict the images (processed)
  - edit folder path in line 146
  - the output file is predictions.txt
  - the accuracy is checked by running check_accuracy.py, where we compare the results to the labels.txt provided
  - side note, because we changed the format of the predictions.txt from separating using '\t' to ',' the check_accuracy.py might not run correctly

## What we achieved:

0.997 on predicting symbols.

* On the training data set that Professor gave us, we achieved 92.7% accuracy when we did modification to each image with clip_brighten.py, and 80.7% accuracy just using clipImage.py

Team members: Andrew Nguyen, Daniel Mints, Abel Seba

## Report:

* Our final report can be found under Final AI Project.pdf