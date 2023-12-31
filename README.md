# Train and Implement an ASL Classification Model Using the Tensorflow Keras API
![C72](https://github.com/brucdeng/ASL-recongition/assets/122509493/50b923b2-a671-4265-b2a3-196c07128598)
<img src="https://github.com/brucdeng/ASL-recongition/assets/122509493/6352f836-b0d6-4b42-95ca-943e3db3ebf5" width="200" height="200">  

This project uses a trained machine learning model to classify various American Sign Language gestures.   

## How to Train  
*Skip this step if you decide to use the pre-trained model included in this repository.*  

Make a copy of Train_ASL_model.ipynb. This is a notebook created on Google Colab that uses the Tensorflow Keras API to train a model to recognize an ASL dataset. Thanks to the large selection of built-in packages that Jupyter Notebook contains, installing anything new is unnecessary.

Run all the code blocks in order. If you would like to use your own dataset, make sure you change the train_folder file path appropriately. The data used to train the model inside this repository is from https://www.kaggle.com/datasets/grassknoted/asl-alphabet. This dataset contains 87,000 images split across 29 categories (3,000 images/category). Each category represented a specific sign language gesture (A-Z, space, delete, nothing).   

After running all the cells, you should have a trained model (.h5 file format) that can classify ASL gestures.  

The code from https://www.kaggle.com/code/raghavrastogi75/asl-classification-using-tensorflow-99-accuracy was helpful for progrmaming this training notebook.

## How to Run
Ensure you have the proper packages installed. 
|Package|Version|
|----------|-------|
|[Tensorflow](https://qengineering.eu/install-tensorflow-2.4.0-on-jetson-nano.html)| 2.4.1 |
|[numpy](https://numpy.org/install/)|1.19.3|
|[opencv-python](https://pypi.org/project/opencv-python/)|4.8.0|
|[python](https://www.python.org/downloads/)|3.6.9|
|[matplotlib](https://matplotlib.org/stable/users/installing/index.html)|2.1.1|
|[Pillow](https://pypi.org/project/Pillow/)|10.0.0|  

Clone this repository. Then run the following command. If you are re-training the model with your own dataset, make sure to replace the file path for the tf.keras.load_model() function inside classifier.py to the correct one for your model. 
```
python classifier.py [FILEPATH TO INPUT IMAGE]
```
For simplicity purposes, there is an input folder containing sample images and available to place your input images. 

In the output folder, you should see output.jpg, which is the model's prediction at the ASL gesture in your input image. Run the following command in the terminal to view output.jpg 
```
display output/output.jpg
```



## Practical Application  
With the existing communication barrier between the deaf community and others, this highly accurate application can help both parties connect with each other easily. Using this program, American Sign Language gestures can now be easily translated to English text, allowing those who do not speak sign language to communicate with the deaf. 
