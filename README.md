#**Finding Lane Lines on the Road** 
<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

This is the first project of the <a href="https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013">Self Driving Car Engineer Nanodegree</a> I am taking part. 

In this project I will detect lane lines in images using Python and OpenCV.

**Dependencies**
<ul>
	<li>OpenCV</li>
	<li>MoviePy</li>
	<li>Numpy</li>
	<li>Matplotlib</li>
</ul>

**Installation**

There is 2 possibilities to run the project:
<ul>
	<li>Install all the dependencies natively</li>
	<li>Run the command in a Conda environment</li>
</ul>

To create the Conda environment, first install <a href="https://www.continuum.io/downloads">Ananconda</a>, then create the environemnt using `environment.yml`: <br />
```
conda env create -f environment.yml
```
Then, activate it: 
```
source activate python3
```