## Advanced Lane Finding

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

[image0]: ./camera_cal/calibration1.jpg "Original"
[image1]: ./output_images/calibration1_undistort.png "Undistorted"
[original_image]: ./test_images/test2.jpg "Original"
[undistorted_image]: ./output_images/undistorted.png "Undistorted"
[sxbinary]: ./output_images/sxbinary.png "sxbinary"
[s_binary]: ./output_images/s_binary.png "s_binary"
[out_img]: ./output_images/out_img.png "Fit Visual"
[lane_detection]: ./output_images/lane_detection.png "Output"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `calibration.py`.  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  

Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. And, I used `cv2.cornerSubPix()` to find more exact corner positions.  

Here is an example image of result of detected chessboard corners.

<img src=./output_images/chessboard_corners.png width="400">

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

| ![][image0] | ![][image1] |
|:--:|:--:|
| input | output |

And, I calculated `Re-projection Error` using `cv2.projectPoints()`.  
Here is a code to calculate `Re-projection Error`.

```python
def calc_reprojection_error(imgpoints, objpoints, mtx, dist, rvecs, tvecs):
	mean_error = 0
	for i in range(len(objpoints)):
		imgpoints_, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i], imgpoints_, cv2.NORM_L2) / len(imgpoints_)
		mean_error += error
	return mean_error
```

### Pipeline (single images)
My pipeline consisted of 5 steps.

1. Distortion correction
1. Color/gradient threshold
1. Perspective transform
1. Detect lane lines
1. Determine the lane curvature

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

| ![][original_image] | ![][undistorted_image] |
|:--:|:--:|
| input | output |

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at `combine_threshold()` in `lane_detection.py`).  

| ![][sxbinary] | ![][s_binary] |
|:--:|:--:|
| gradient threshold | color threshold |

Here's an example of my output for this step.

<img src=./output_images/binary.png width="400">

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 1 through 8 in the file `lane_detection.py`.  

The `perspective_transform()` function takes as inputs an image (`binary`), as well as source (`src_points`) and destination (`dst_points`) points.  I chose the hardcode the source and destination points in the following manner:

```python
def get_warp_points():
	corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
	new_top_left  = np.array([corners[0, 0], 0])
	new_top_right = np.array([corners[3, 0], 0])
	offset = [50, 0]
	src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
	dst_points = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])
	return src_points, dst_points
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| (253, 697)      | (303, 697)        | 
| (585, 456)      | (303, 0)      |
| (700, 456)      | (1011, 0)      |
| (1061, 690)     | (1011, 690)        |

I verified that my perspective transform was working as expected by drawing the `src_points` and `dst_points` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

|<img src=./output_images/binary.png> | <img src=./output_images/warped_binary.png> |
|:--:|:--:|
| input image | warped image |

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This processing consisted of 3 steps.

1. Find starting point for where to search for the lines
    - Calculate a histogram of the bottom half of the image using `np.sum()`.
    - Find two highest peaks from histogram.
1. Track curvature using sliding windows
    - From starting point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
1. Fit a polynomial
    - Fit a second order polynomial to each using `np.polyfit()`.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

- Green rectangles
  - sliding windows
- Yellow lines
  - polynomial lines

![][out_img]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of curvature using [this formula](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). I did this processing at `get_curvature()` in `lane_detection.py`.  

This processing consisted of 3 steps.

1. Define conversions in x and y from pixels space to meters
1. Define y-value where we want radius of curvature
1. Calculation of R_curve (radius of curvature)

Here is a code to calculate the radius of curvature.

```python
def get_curvature(leftx, lefty, rightx, righty, ploty, image_size):
	y_eval = np.max(ploty)

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	scene_height = image_size[0] * ym_per_pix
	scene_width = image_size[1] * xm_per_pix

	left_intercept = left_fit_cr[0] * scene_height ** 2 + left_fit_cr[1] * scene_height + left_fit_cr[2]
	right_intercept = right_fit_cr[0] * scene_height ** 2 + right_fit_cr[1] * scene_height + right_fit_cr[2]
	calculated_center = (left_intercept + right_intercept) / 2.0
	lane_deviation = (calculated_center - scene_width / 2.0)

	return left_curverad, right_curverad, lane_deviation
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step at `draw_lanes()` in `lane_detection.py`.  
This processing consisted of 3 steps.

1. draw the polygon of lane area using `cv2.fillPoly()`
1. warp lane area from birdview to original view using `cv2.warpPerspective()`
1. overlay lane area to original image using `cv2.addWeighted()`

Here is a code to plot lane area onto image.

```python
def draw_lanes(img, M_inv, left_fitx, right_fitx, ploty, left_curvature, right_curvature, lane_deviation):
	color_warp = np.zeros_like(img).astype(np.uint8)
	pts_left   = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right  = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	lane_image = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
	result = cv2.addWeighted(img, 1.0, lane_image, 0.3, 0)

	# draw text
	curvature_text = 'Curvature: Left Lane = ' + str(np.round(left_curvature, 2)) + ', Right Lane = ' + str(np.round(right_curvature, 2))
	font = cv2.FONT_HERSHEY_COMPLEX
	cv2.putText(result, curvature_text, (30, 30), font, 1, (0,255,0), 2)
	deviation_text = 'Lane deviation from center = {:.2f} m'.format(lane_deviation)
	cv2.putText(result, deviation_text, (30, 60), font, 1, (0,255,0), 2)

	return result
```

Here is an example of my result on a test image:

![][lane_detection]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I used [project_video.mp4](project_video.mp4) to evaluate my pipeline.  
Here's a [link to my video result](./output_images/project_video.mp4).

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### Faced problems
I tried `challenge_video.mp4`. As a result, I was faced with some problem

In this case, my pipeline judges cracks of road as lanes.

<img src=./output_images/NG1.png width="400">

This cracks has the edge of vertical direction. This is the cause of failure to track the curvature using sliding windows.

<img src=./output_images/NG1_out_img.png width="400">

In this case, my pipeline can not fit the left lane properly.

<img src=./output_images/NG2.png width="400">

I checked the result to track the curvature. As a result, I think that information for fitting is insufficient.

<img src=./output_images/NG2_out_img.png width="400">

And, I checked binary image. I found that shadowed lane region can not be extracted.

<img src=./output_images/NG2_binary.png width="400">

#### Identification of potential shortcomings
I think that my current pipeline has shortcoming in the following condition.

1. Outliers
    - Current pipeline does not remove outliers. So, current pipeline is weak to outliers.
    - Current pipeline uses only current frame. So, current pipeline is weak to outliers.
    - Current pipeline uses color information to extract lane. But, lane can not be extracted correctly when illumination condition changes.

#### Suggestion of possible improvements
I suggest improvements to overcome mentioned shortcomings.

1. Sanity Check
    - Current pipeline found some lines. Before moving on, pipeline should check that the detection makes sense.
1. Smoothing
    It can be preferable to smooth over the last n frames of video to obtain a cleaner result.

---

## Reference
- [OpenCV: Camera Calibration](https://docs.opencv.org/3.4.2/dc/dbb/tutorial_py_calibration.html)
- [Radius of Curvature](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)
