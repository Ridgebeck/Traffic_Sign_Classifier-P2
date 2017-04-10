# **Traffic Sign Recognition** 

## **Build a Traffic Sign Recognition Project**
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data_graph.jpg "Training Data"
[image2]: ./sample_sign_pprocess.jpg "Preprocessing"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

#### Here is a link to my [project repository on GitHub](https://github.com/Ridgebeck/Traffic_Sign_Classifier-P2)

### Data Set Summary & Exploration

#### 1.Basic summary of the data set

The code for this step is contained in the second code cell of the IPython notebook / html file.  

I used numpy methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of test set is 12,630
* The shape of a traffic sign image is 32, 32, 3 - 32x32 pixels with RGB values
* The number of unique classes/labels in the data set is 43

#### 2.Exploratory visualization of the dataset

The code for this step is contained in the second and third code cell of the IPython notebook / html file.  

Here is an exploratory visualization of the training data set. It shows the distribution of the pictures over the different labels. You can see that there is quite some difference between the 43 different labels, whcih means that some of them get trained much more often than others and the algorithm is more likely to recognize similiar images.




![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the LeNet algorythm was designed for black and white images (text) and a test with all of the other parameters staying the same showed an increase in accuracy.
As a last step, I normalized the image data because this minimizes the variations in the picture and makes it easier for a CNN to optimize the weights during backpropagation.

Here is an example of a traffic sign image before and after grayscaling + normalization:


![alt text][image2]



#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 3. The final model architecture was the LeNet architecture that was modified with a dropout function applied to the first fully connected layer

The code for my final model is located in the sixth cell of the ipython notebook / html file. 

My final model consisted of the following layers:

| Layer         	      	|     Description	        				                 	| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 grayscaled normalized image   						 	| 
| Convolution 5x5      	| 1x1 stride, valid padding, outputs 28x28x6   	|
| Relu			             	 | relu function of first convolution layer						|
| Max pooling	      	   | 2x2 stride, valid padding, outputs 14x14x6 			|
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16   |
| Relu                  | relu function of second convolution layer	    |
| Max pooling				       | 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten    				       | Flatten the input of 5x5x16 to output 400     |
| Fully connected       | input 400, output 120                         |
| Relu       				       | relu function of first fully connected layer	 |
| Dropout    				       | dropout function applied                      |
|	Fully connected       |	input 120, output 84 					              						|
| Relu       				       | relu function of second fully connected layer	|
|	Fully connected       |	input 84, output 43  					              						|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			              |     Prediction	        					                  | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign           		| Stop sign   								                         	| 
| U-turn     		        	| U-turn 									                             	|
| Yield					            | Yield										                              	|
| 100 km/h	           		| Bumpy Road					 			                          	|
| Slippery Road		      	| Slippery Road      				                    			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
