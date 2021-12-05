## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, my goal was to write a software pipeline to identify the lane boundaries in a video,
but the main output or product I came up with was a detailed [writeup](writeup.md) of the project.

The Project
---

The achieved goals / steps of this project are the following:

* Computing the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Applying a distortion correction to raw images.
* Using color transforms, gradients, etc., to create a thresholded binary image.
* Applying a perspective transform to rectify binary image ("birds-eye view").
* Detecting lane pixels and fit to find the lane boundary.
* Determining the curvature of the lane and vehicle position with respect to center.
* Warping the detected lane boundaries back onto the original image.
* Putting out visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called [camera_cal](camera_cal).  The images in [test_images](test_images) were used for testing the pipeline on single frames.

Examples of the work were stored in the folder called [output_images](output_images), and they include a description in my writeup for the project of what each image shows.
The video called `project_video.mp4` is the video this pipeline works well on.
