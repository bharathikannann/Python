# `Machine Learning and Deep Learning`

## Table of contents

- [Machine Learning](#machine-learning)
	- [Machine Learning Algorithms](#machine-learning-algorithms)

- Artificial intelligence (AI) is the broad science of mimicking human abilities, machine learning is a specific subset of AI that trains a machine how to learn.

## Machine Learning

> - **Machine learning is a field of study that gives computers the ability to learn without being explicitly(clearly) programmed**.
- Machine learning techniques are used to automatically find the valuable underlying patterns within complex data that we would otherwise struggle to discover. The hidden patterns and knowledge about a problem can be used to predict future events and perform all kinds of complex decision making.

- Tom Mitchell provides a more modern definition: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.


### Machine Learning Algorithms

- **Supervised Learning** 
- **Unsupervised Learning**
- **Reinforcement Learning**

## Supervised Learning

- Supervised learning is when the model is getting trained on a labelled dataset.
- Labelled dataset is one which have both input and output parameters.

#### Example
- Given data about the area of houses and their price, try to predict their price for new area.

### Types of supervised learning
- `Regression`
- `Classification`

#### Regression
- Here we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.
- Given a picture of a person, We have to predict his/her age(Continuous).

#### Classification
- Here we are trying to map input variables into discrete categories. 
- Given a picture of a person, We have to predict their gender (Male/Female) (Discrete).

### Linear Regression

- Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data.

- Training Dataset
	> - **X**   **Y**
	> - 2500    550000
	> - 2700    550000
	> - 3000    565000

- **Hypothesis Function**

- Our hypothesis function
	- `Prediction ​= weight * x + bias`
- Example
	- Prediction = weight * (2800)  + bias​

### Cost Function

- Accuracy of our hypothesis function can be measured by using cost function.

- A Cost function basically tells us how good our model is at making predictions for a given value of weight and bias.

- **Formula**
	> - Cost(weight , bias) = `𝟏/𝟐𝒎 ∑(𝒊=𝟏)^𝒎[[(𝑯𝒚𝒑𝒐𝒕𝒉𝒆𝒔𝒊𝒔(𝒙_𝒊)−𝒚_𝒊)]^𝟐 ]`
	- `m` - Number of training examples.
	- `𝐻𝑦𝑝𝑜𝑡ℎ𝑒𝑠𝑖𝑠 𝑥_𝑖 − 𝑦_𝑖` - Difference of Predicted and Original value.

- This takes an average of all the results of the hypothesis to the actual output y's.
- Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least.

### Our Goal
- `minimize(cost(weight,bias))`

### Gradient Descent
- Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) that minimizes a cost function.
> - There are two things that you should know to reach the minima.
> 	- `which way to go and how big a step to take. `

- We can use gradient descent to do this with the use of derivatives effectively.
- A derivative is the slope of a graph at any given point and it will give us the direction to move towards.

- **Learning Rate(∝)**
	- The size of steps taken to reach the minimum is called learning rate.
	- If we use higher learning rate it will move larger steps but there is a risk of overshooting the minima.
	- If we use lower learning rate it will take more steps to converge.
	- Try Experimenting with it
		- [Source](https://developers.google.com/machine-learning/crash-course/fitter/graph)

- **Algorithm** 

	> - weight := weight - learning_rate * differentiation of cost with respect to weight
	> - bias := bias - learning rate * differentiation of cost with respect to bias

	- if the derivative(slope) is positive then subtracting it will decrease the weight so it will move towards minima.
		- `w:=w-learning_rate * (positive slope) here w will decrease.`
	- if the derivative(slope) is negative the subtracting it will increase the weight si it will move towards minima. 
		- `w:=w-learning_rate * (negative slope) here w will increase.`
	- same as for bias

