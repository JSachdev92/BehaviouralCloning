# **Behavioral Cloning** 

## Jayant Sachdev

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture-nvidia.png "NVidia CNN Architecture for Autonomous vehicles"
[image2]: ./Training_Results/Center.png "Center Straight"
[image3]: ./Training_Results/Left.png "Left Straight"
[image4]: ./Training_Results/Right.png "Right Straight"
[image5]: ./Training_Results/Center_Curve.png "Center Straight"
[image6]: ./Training_Results/Left_Curve.png "Left Straight"
[image7]: ./Training_Results/Right_Curve.png "Right Straight"
[image8]: ./Training_Results/Right_Curve.png "Right Straight"
[image9]: ./Training_Results/Recovery.png "Recovery"
[image10]: ./Training_Results/Bridge.png "Bridge"
[image11]: ./Training_Results/Road_Edge.png "Road Edge"
[image12]: ./Train_Val_errLoss_final.png "Validation and Training Loss"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Project Summary
In this project, i utilized deep learning algorithms to train a vehicle to drive autonomously around a racetrack in a simulator. I first collected data in the simulator; then utilized a convolutional neural network (CNN) to learn the images and the associated steering angles. Then used the learned model to drive the vehicle autonomously in the simulator.  

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Since the goal of the project was to drive a car autonomously around a track in a simulator, i levereged the CNN architecture designed by NVIDIA for autonomous vehicles. The architecture is explained in detail in this [paper] (https://arxiv.org/pdf/1604.07316v1.pdf) and illustrated below:

![alt text][image1]

Using this architecture as a baseline, i first normalized the images using a lambda layer, then cropped the top 60 and the bottom 25 pixels to focus on the key aspects of the image. 

```
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,25),(0,0))))
```

I also added a dropout layer before the fully connected layers to help prevent overfitting of the model.

```
model.add(Dropout(0.5, noise_shape=None, seed=None))
```

#### 2. Attempts to reduce overfitting in the model

There were several design decisions taken to reduce overfitting in the model. To begin with, the data was trained and validated on separate data sets. Looking at `model.py` lines 113-122, you can see that i utilized the train_test_split function to split off 20% of the data for validation purposes.  

Furthermore, a dropout layer was added just before the 3 fuller connected layers. I experimented with adding a 2nd dropout layer after the first convolutional layer in addition to the dropout layer before the fully connected layers, but found that doing so resulted in underfitting and poor performance. As a result i just utilized the one dropout layer. 

When training the model, i noticed that the validation loss would start to stabalize around 13-15 epochs. As a result i set the model to train for 15 epoch's only. In addition, i saved the parameters for the model with the best validation loss to assist with the overfitting. 

The model validation and training loss was also monitored to ensure that the loss were in the same ballpark and the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. As suggested in the project introduction and overview, a Mean Square Error loss function was used. In total, the validation and training loss was observed during the training process, and it was found that the validation loss would start to stabalize at around 13-15 epochs. As such, the final model was trained for 15 epochs. Furthermore, only the model with the lowest validation loss was saved to ensure that the best model is utilized.  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
