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
[image4]: ./web_images/1.jpg "Traffic Sign 1"
[image5]: ./web_images/2.jpg "Traffic Sign 2"
[image6]: ./web_images/3.jpg "Traffic Sign 3"
[image7]: ./web_images/4.jpg "Traffic Sign 4"
[image8]: ./web_images/5.jpg "Traffic Sign 5"
[image9]: ./predictions/bar_1.jpg "Bar Graph Image 1"
[image10]: ./predictions/bar_2.jpg "Bar Graph Image 2"
[image11]: ./predictions/bar_3.jpg "Bar Graph Image 3"
[image12]: ./predictions/bar_4.jpg "Bar Graph Image 4"
[image13]: ./predictions/bar_5.jpg "Bar Graph Image 5"


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


#### 2. The final model architecture was the LeNet architecture that was modified with a dropout function applied to the first fully connected layer

The code for my final model is located in the sixth code cell of the IPython notebook. 

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
 


#### 3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh code cell of the IPython notebook.

I left most of the parameters at the original values of the LeNet architecture, such as:

* 10 epochs
* batch size of 128
* mu = 0
* sigma = 0.1

I added a dropout function with a keep probability of 0.5 to the first fully connected layer to make the model more robust and better suitable against overfitting. This allowed me to double the learning rate from 0.001 to 0.002.


#### 4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results after 10 epochs were:

* validation set accuracy of 94.4%
* test set accuracy of 91.1%

The following parameters were modified to find the best solution in an iterative approach:

* learning rate
* dropout rate in fully connected layers
* pre-processing parameters (grayscale, normalization)

The parameters were changed individually and the validation accuracy was taken to rate the model. The highest validation accuracy was reached with the grayscaled and normalized pictures, a learning rate of 0.002 and a dropout rate of 0.5 in the first fully connected layer. The dropout with various keep probabilities (0.5, 0.75, 0.9) was tested in the second fully connected layer as well, but did not bring any advantages to the model.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


The last image might be difficult to classify because it is slightly tilted/angled and the training set was consisting of pictures that were shot straight from the front. To make the algorithm more robust for pictures which are shot from different angles more pre-processing of the training data would have been necessary - such as tilting, rotating, distorting, etc.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			              |     Prediction	        					                  | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign (14)      		| Stop Sign (14) 					                         	| 
| 30 km/h (1)		        	| 30 km/h (1)		 		                             	|
| Road Work (25)        | Road WOrk (25)                               	|
| No Passing (9)	     		| No Passing (9)	 			                          	|
| 80 km/h (5)          	| 30 km/h (1)	       				                    			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The false prediction was as expected the picture that was shot from an angle.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Here are the top 5 predictions for every image:

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

The first picture was correctly predicted as a STOP (14) sign, but the algorithm was also thinking that it could have been a KEEP RIGHT (38). Both probabilities were pretty close with 46% and 41%. Picture 2, 3, and 4 were pretty clear with the first choice always having more than 87% and all images were predicted correctly. The last picture was predicted incorrectly (30km/h (1) insted of 80 km/h (5)) with a fairly high confidence of 79%. The second choice would have been the right one with 7%.

The accuracy is lower on the images from the web as expected as the images are varying more than the ones from the original training and test set. To achieve a higher accuracy further pre-processing of the original training set could have been done, such as rotation, zoom, flipping of the images etc. to make the algorithm more robust for different inputs.

