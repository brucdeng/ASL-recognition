# Train and Implement an ASL Classification Model Using the Tensorflow Keras API
![C72](https://github.com/brucdeng/ASL-recongition/assets/122509493/50b923b2-a671-4265-b2a3-196c07128598)
<img src="https://github.com/brucdeng/ASL-recongition/assets/122509493/6352f836-b0d6-4b42-95ca-943e3db3ebf5" width="200" height="200">  

This project uses a trained machine learning model to classify various American Sign Language gestures.

## How to Train  
*Skip this step if you decide to use the pre-trained model included in this repository.*  

Make a copy of Train_ASL_model.ipynb. This is a notebook created on Google Colab that uses the Tensorflow Keras API to train a model to recognize an ASL dataset. 

Run all the code blocks in order. If you would like to use your own dataset, make sure you change the train_folder file path appropriately. The data used to train the model inside this repository is from https://www.kaggle.com/datasets/grassknoted/asl-alphabet.  

After running all the cells, you should have a trained model (.h5 file format) that can classify ASL gestures.  

The code from https://www.kaggle.com/code/raghavrastogi75/asl-classification-using-tensorflow-99-accuracy was extremely helpful for progrmaming this training notebook. Credit goes to Raghav Rastogi for inspiring me to create this model!  

## How to Run
Ensure you have the proper dependencies and libraries installed. 
|Package|Version|
|----------|-------|
|[Tensorflow](https://qengineering.eu/install-tensorflow-2.4.0-on-jetson-nano.html)| 2.4.1 |
|numpy|1.19.3|
|opencv-python|4.8.0|
|python|3.6.9|
|matplotlib|2.1.1|
|Pillow|10.0.0|  

Clone this repository. Then run the following command. 
```
python classifier.py [FILEPATH TO INPUT IMAGE]
```
For simplicity purposes, there is an input folder containing sample images and available to place your input images. 

In the output folder, you should see output.jpg, which is the model's prediction at the ASL gesture in your input image.  

## Demonstration
Video Link: 
