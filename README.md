# Machine Learning Experiments
This repository showcases some machine learning models I've played around with using sample data sets from Kaggle.
There are two folders in this repository, sine and predictive maintenance. They use different models and data sets, where sine
is simply the sine function and the predictive maintenance holds machine learning features and a boolean column for broken.

### Sine Prediction
<img src="https://github.com/AndyLc/ml-experiments/blob/master/sine/sine.png" width=400>
With a generated data set of the sine function, as shown above, our goal is to learn it and be able to predict the y values over the next
x values. 

In order to predict the next few timesteps (x values), we will use a supervised learning algorithm with a standard neural network. 
Because the standard neural network does not have memory of the past inputs and can only predict the next y values given a set of x values,
to turn the time series data into a supervised learning data set, we set our x values to the past few y values in order to predict the next
y value.

| Theta         |
| :-----------: |
| t_0           |
| t_1           |
| t_2           |
| ...           |

Our table originally is in this format, where t_0 is the y value of the sine function at time 0 and t_x is the y value of the sine function at time x.

| Theta         | Theta         | Theta  |
| :-----------: |:-------------:| :-----:|
| t_0           | t_1           | t_2    |
| t_1           | t_2           |   t_3  |
| t_2           | t_3           |    t_4 |

After changing our data format into a supervised data format, we can now use the first 2 columns to predict the 3rd. 
In the ipython notebook I use the first 6 columns to predict the 7th, as to me that yielded the best results.

After training the model, which is a Keras Sequential with 2 Dense layers in the hidden layer (specifics can be found in the ipython notebook),
I found an error calculated using mean squared error of 6.838e-07 in predicting the y value given the exact y values in the last 6 timesteps.

<img src="https://github.com/AndyLc/ml-experiments/blob/master/sine/sine-1-timestep.png" width=400>

Our model worked very well in learning how to predict the next timestep given the correct values of the last few timesteps.
However, in real life we often want to predict more timesteps into the future than 1. In this case we first predict timestep t+1,
and we predict timestep t+2 using our prediction of t+1. Therefore, overtime the error will build up since the error in timestep t+1 will
cause additional error in predicting the correct value in timestep t+2. 

<img src="https://github.com/AndyLc/ml-experiments/blob/master/sine/sine-long-prediction.png" width=400>

### Predictive Maintenance
One application of machine learning is to predict when some kind of machine will break. This would be very useful for airplanes,
as predicting when an airplane will need maintenance would be critical in passenger safety. I explored this concept
in the Predictive Maintenance folder, where I used a dataset of a machine shown below.

<img src="https://github.com/AndyLc/ml-experiments/blob/master/predictive_maintenance/pm-data.png" width=400>

First we want to be able to predict whether or not a machine is broken given the features of temperature, lifetime, etc. 
This is a classification problem, so I used the Naive Bayes training algorithm. I processed the data through a variety of steps listed
in the ipython notebook, and eventually got a result of 0.8367% accuracy in predicting whether a machine was broken or not. 

However, the more useful information is to determine the number of days until the machine will break. I had to add a new column for this, 
timeTilBroken, which takes data from our given columns and calculates the time until the machine is broken. The new data is shown below.

<img src="https://github.com/AndyLc/ml-experiments/blob/master/predictive_maintenance/processed-pm-data.png" width=400>

In order to predict this value, we use a neural network because this is no longer a classification problem.
The results seem like they are accurate, with our predictions being off by one or two days usually. However, I had to make many assumptions
in order to come to this result within processing the data, so the results may be deceptively accurate. Some of these assumptions involved
the general lifetime of a machine, in which case I picked a fixed number and added an aritifical uniform error.

Thank you for reading my experiments with machine learning and please visit 
my portfolio at https://andylc.github.io/ if you'd like to reach out to me.
