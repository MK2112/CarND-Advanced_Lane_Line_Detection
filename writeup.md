## Writeup  - Project 2: Advanced Lane Finding
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup/undistort_distorted.png "Undistortion"
[image2]: ./test_images/test5.jpg "Road Original"
[image7]: ./writeup/undistortion.png "Original vs. Undistorted Road"
[image3]: ./writeup/thesholding.png "Binary Example"
[image4]: ./writeup/source_destination.png "Source And Destination Points Example"
[image8]: ./writeup/warp_example.png "Warp Example"
[image5]: ./writeup/windows.png "Fit Visual"
[image6]: ./writeup/polyfit.png "Polyfit"
[image9]: ./writeup/color_channels.png "Chosen Color Channels"
[image10]: ./output_images/test1.jpg "Lane Detection on Image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

Well, this is it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step contained in the third and fourth code cells of the Jupyter notebook located in "./P2.ipynb".

The process starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. I assume the chessboard to be fixed on the (x, y) plane at z=0, so that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The `mtx` ( camera matrix) and `dist` (distortion coefficient) are then - after everything is done and finished - saved to a file called "calibration_pickle.p". This proved to be a great thing while testing out the pipeline, as the camera calibration's results now were saved and did not need to be re-calculated every time.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I applied the distortion correction to one of the test images like this one:
![alt text][image2]

I wrote a function called `cal_undistort`. It accesses the "calibration_pickle.p" file previously created by the `cal_camera` function and reads out the variables `mtx` and `dist`. These are then applied as parameters to OpenCV's `cv2.undistort` function. This results in an image being undistorted just like the chessboard images were, thereby removing e.g. image warping influences like lens curvature and the like.
The resulting undistorted image is now returned. It now looks like this. I chose this image as an example because the removal of a certain fish eye effect can be seen pretty well on the white Lexus at the right edge. 
![alt text][image7]

It's interesting to see such a seemingly small change have such a high impact on the overall system's accuracy.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

As can be seen in the Jupyter Notebook's subsection 'Experimenting: Image correction' this phase involved a lot of experimentation. First I looked for a test image from the test image data set that would show the most diverse lighting possible. I then wanted to use this to determine the color channels that would represent the lanes most clearly. I thought this image had the most interesting lighting characteristics:
![alt text][image2]

I tried various combinations of color and gradient thresholds to generate a binary image. To me HLS's saturation channel and HSV's value channel seemed the most promising. 
![alt text][image9]

Playing around with the thresholds this is what I came up with then:

`s_thresh = (180, 200)
v_thresh = (210, 255)`

Now gradient detection was applied on HSV's value channel. the color channels and the gradients combined were then thesholded. The sobel-thesholds looked like this:

`sobel_x_thresh = (40, 100)`

But at this point I was not really happy with the result. It captured the lane lines and did so pretty well, but it still detected
what I would call 'false lane pixels', meaning pixels that met all the thresholds but were not actually displaying parts of the lane lines.
I solved this problem by applying a light gaussian blur to the input image.

The thresholding steps and results per step can be seen in the notebook's subsection 'Experimenting: Image correction'. The findings of this experimental phase later were applied to the pipeline which can be found in the subsection 'Bringing it all together: Functions of a pipeline'.

In total: I undistort the image, then a light blur is applied to it. Only I continue to isolate color channels and apply gradient detection.

Here's an example of my output for this step for an actual test image:
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

All functions regarding single image lane detection and illustration were placed in a subsection called 'Bringing it all together: Functions of a pipeline' in the Jupyter notebook.

After having created the thresholded binary image described for step 2, I applied region masking to isolate the region of interest. Only then did I continue to apply a perspective transform.

The code for my perspective transform includes a function called `birds_eye()`.  This function only takes as input an image (`img`) and internally calculates source (`src`) and destination (`dst`) points from hard-coded point-values I found through experimenting.

This is what it looks like:

```python
    # Hard-coded, found through experimenting
    x_in = [575, 200, 1120, 730]
    y_in = [460, 690, 690, 460]

    x_out = [220, 200, 1060, 1065]
    y_out = [0, 720, 720, 0]
    
    src = np.float32([[x_in[0],y_in[0]],[x_in[1],y_in[1]],[x_in[2],y_in[2]],[x_in[3],y_in[3]]])
    dst = np.float32([[x_out[0],y_out[0]],[x_out[1],y_out[1]],[x_out[2],y_out[2]],[x_out[3],y_out[3]]])
```

The relationship between source and destination points looks like this:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 220, 0        | 
| 200, 690      | 200, 720      |
| 1120, 690     | 1060, 720     |
| 730, 460      | 1065, 0       |

I found these values for the `src` and `dst` by moving points on a test image until the source points were at the beginning and end of the lanes and their warped counterparts were at the top and the bottom of the image.

This is what it looked like:

![alt text][image4]

Applied to this test image:

![alt text][image2]

This was the result:

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Only after creating a thresholded binary image, region masking it and applying the 'bird's eye warp' to it did I start the search for lane line fitting polynomial functions.
I start out by creating a horizontal histogram for the lower half of the image. I cut it into two halfs, a right one and a left one. Each of these partial histograms is then searched for a peak. I assume that this peak shows the horizontal position of, depending on right or left histogram, the right or left lane. Thoses peaks are saved as `leftx_base` and `rightx_base` for later use.
I continue to apply (by default 9) windows per left and right lane for the sliding window lane detection. All of these windows are positioned so that either `leftx_base` or `rightx_base` becomes their vertical center. The rectangles get their respective heights from dividing the image's full height across the amount of windows per lane (in this case 9 by default) and their widths from a margin (by default 100 pixels) applied to the left and the right side of the known center (known from the histogram.

Then per window we look out for a white pixel cluster above a certain size (white pixels are the lane pixelstherefore known lane pixels) is searched. If found away from the current center, the window is moved accordingly to now hold this cluster's center as new center point.
This approach results in windows being centered each and individually around actual lane pixels. From this information we now can derive both lanes shapes.

![alt text][image5]

As each window now has a center positioned inside the biggest reachable pixel cluster the windows for one lane all found points on a polynomial function showing the course of the lane. These 'good indicators' are gathered and a polynomial gets fit to these found points per lane. Thus a fitting function is found.

![alt text][image6]

The function `find_lanes` can be found in the Jupyter Notebook's subsection titled 'Bringing it all together: Functions of a pipeline'.
I've pasted it here aswell:

```
# interconnecting pixel clusters that form a line
def find_lanes(img, nwindows=9, margin=100, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    ym_per_pixel = 30/720 
    xm_per_pixel = 3.7/700 
    
    left_fit_m = np.polyfit(lefty*ym_per_pixel, leftx*xm_per_pixel, 2)
    right_fit_m = np.polyfit(righty*ym_per_pixel, rightx*xm_per_pixel, 2)
    
    radius, offset = get_radius_and_offset(left_fit_m, right_fit_m, ym_per_pixel, xm_per_pixel)

    # Highlight lane pixels and draw fit polynomials
    lane_pixel_img = np.dstack((img, img, img))*255
    lane_pixel_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    lane_pixel_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, radius, offset, out_img

```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the Jupyter Notebook's subsection titled 'Bringing it all together: Functions of a pipeline' I defined a function called `get_radius_and_offset`.

```
# Calculating the radius in meters for both fitted functions
# Radius is mean of radius of both lanes
# Offset indicates how far camera is away from center
def get_radius_and_offset(left_fit, right_fit, y_m_pixel, x_m_pixel):
    l_poly_radius = ((1 + (2 * left_fit[0] * 720 * y_m_pixel + left_fit[1]) ** 2) ** (3 / 2)) / np.abs(2 * left_fit[0])
    r_poly_radius = ((1 + (2 * right_fit[0] * 720 * y_m_pixel + right_fit[1]) ** 2) ** (3 / 2)) / np.abs(2 * right_fit[0])

    left_lane = left_fit[0] * (720 * y_m_pixel) ** 2 + left_fit[1] * 720 * y_m_pixel + left_fit[2]
    right_lane = right_fit[0] * (720 * y_m_pixel) ** 2 + right_fit[1] * 720 * y_m_pixel + right_fit[2]

    radius = np.mean([l_poly_radius, r_poly_radius])
    offset = [640 * x_m_pixel - np.mean([left_lane, right_lane]), right_lane - left_lane]
    return radius, offset
```

I calculate the radius per fitted lane polynomial and translate into real-world units. The 'image radius' then is calculated from the mean of both left- and right-lane radius. The offset then is calculated based on the center between both left and right lane polynomial. This is what I view as the street lane center, so any difference in image center from this street lane center is identified as offset.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step with the two code blocks in the Jupyter Notebook's subsection 'Bringing it all together: Functions of a pipeline'. The second code block takes advantage of the functions defined in the first code block. Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline works pretty well for the given test image set and the project video. Certain lighting conditions though still cause some slight wobbliness.
Now a problem with this pipeline would be e.g. different lighting conditions caused for example by weather or daytime. Another factor would be lane width or existence of lane markings at all. They are expected to be there on the road and within a certain part of the image. If a car is in front of the autonomous vehicle. For example in stop and go traffic a car right ahead might disturb the lane detection.

There are multiple ways to improve the pipeline. One way to make it more robust would be to increase the set of test images. One could add more images of different quality road circumstances. The region of interest should not be set within hard-coded boundaries. The color optimization and gradient detection should be more dynamic, meaning that a color channel or combinations of color channels should be chosen by the pipeline instead of hard-coded by a developer.