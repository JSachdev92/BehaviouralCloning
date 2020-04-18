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

---
### Project Summary
In this project, i utilized deep learning algorithms to train a vehicle to drive autonomously around a racetrack in a simulator. I first collected data in the simulator; then utilized a convolutional neural network (CNN) to learn the images and the associated steering angles. Then used the learned model to drive the vehicle autonomously in the simulator.  

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Since the goal of the project was to drive a car autonomously around a track in a simulator, i levereged the CNN architecture designed by NVIDIA for autonomous vehicles. The architecture is explained in detail in this [paper](https://arxiv.org/pdf/1604.07316v1.pdf) and illustrated below:

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

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and additional data on the bridges and section of road where i noticed the vehicle struggling to stay centered, such as where there was no lane marking on one side. I also made sure that the majority of training data was focused on smooth driving. I noticed if there was too much recovery driving, the baseline performance of the vehicle was low and additional smooth driving data was needed. 

For details about how I created the training data, see the next section. 


#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize an architecture that has been validated for autonomous vehicle applications. It was the reason for selecting the NVIDIA CNN designed specifically for end-to-end learing for self-driving cars.

I utilized the optional data as a starting point and trained the data for 20 epochs. In order to gain insight into the model i split the data into a training and validation set with 20% of the data being split into validation data. I noticed that the model was starting to overfit after 6-7 epochs, so i added a dropout layer before the fully connected layers.

During testing, i noticed that the vehicle veered off track in corners and on the bridge, so i collected more data for recorvery and for driving in unique sections, such as on the bridge, or on the section with lane line on one side. 

I still noticed that the training loss kept decreasing while the validation loss stabalized after 10 epochs, so i added another dropout layer after the first convolutional layer. 

However, i noticed that the performance kept on degrading rather than improving. After collecting multiple sets of new data and training on that, i made a few realizations. firstly, the ratio of data for each situation is important. I had collected too much data focusing on recovering in various scenario's that it dwarfed the normal driving data and the vehicle would behave poorly as a result. In addition, i realized that the 2nd dropout layer was causing underfitting, which was not noticed in the data, but was clearly visible in the simulator as no amount of new data would resolve the performance issues. 

As a result, i removed the dropout layer after the first convolutional layer and kept the dropout layer before the fully connected layers. Also, i collected a competely new data set that contained 3 laps of smooth driving out of which 1 lap was recorded going the other way. In addition to the 3 laps, about 30% additional data was recorded doing recovery maneuvers and regular driving in parts of the track that were unique, like the bridge and sections with only one lane line.  

I noticed that the validation loss and training loss was much lower than the other datasets to begin and there was no indication of overfitting in this model.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 6. Final Model Architecture

The final model architecture, coded in `model.py` lines 54-72, followed the basic structure of NVIDIA CNN illustrated above and consisted of the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 images   							| 
| Normalization         	| normalize pixel data to [-1,1] | 
| Cropping         	| 2D cropping, 60 pixels from the top and 20 pixels from the bottom | 
| Convolution 5x5    | 	 2x2 stride, 24 depth, Relu activation |
| Convolution 5x5    | 	 2x2 stride, 36 depth, Relu activation |
| Convolution 5x5    | 	 2x2 stride, 48 depth, Relu activation |
| Convolution 3x3    |   non-strided, 64 depth, Relu activation|
| Convolution 3x3    |   non-strided, 64 depth, Relu activation|
| Flatten					|	1164  Outputs	|
| Dropout		|		
| Fully connected		| 100 Outputs	|
| Fully connected		| 50 Outputs	|
| Fully connected		| 10 Outputs	|
| Fully connected		|  Steering Angle Output  |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using smooth center lane driving. Here are some example images of center lane driving:

| Center Image         		|     Left Image 	        					|  Right Image |
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:| 
|![][image2]|![][image3]|![][image4]|
|![][image5]|![][image6]|![][image7]|

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when getting close to a lane edge. An example of recovery can be seen below:

![alt text][image8]


To augment the data sat, I also flipped images and used the left and right side images with correction factors as this would assist with creation of the data, assist with recovery and normalize the data since it normalizes the number left and the right turn data.

Initially, i utilized the provided data set and added additional data to that. I made the realization that in the default dataset, there is a lot of data where there is no steering angle. In addition, i had collected almost triple the default dataset in recovery data. As a result the vehicle model did not have enough smooth driving data to complete the track. 

I utilized this learning to collect data where i used the mouse input to always have a steering angle during recording, especially during smooth driving data. Also, i limited the recovery and special scenario data to approx 30% of the the total dataset. 

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
