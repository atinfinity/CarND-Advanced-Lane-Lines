import numpy as np
import cv2
import pickle
import glob
import argparse

def load_camera_param(param_file):
	f = open(param_file, 'rb')
	dist_pickle = pickle.load(f)
	mtx  = dist_pickle['mtx']
	dist = dist_pickle['dist']
	f.close()
	return mtx, dist

def convert_hls(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def undistort(img, mtx, dist):
	return cv2.undistort(img, mtx, dist, None, mtx)

def combine_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
	img = np.copy(img)
	# Convert to HLS color space and separate the V channel
	hls = convert_hls(img)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]

	# rgb thresholding for yellow
	lower_rgb_yellow = np.uint8([  0, 180, 225])
	upper_rgb_yellow = np.uint8([170, 255, 255])
	mask_rgb_yellow = cv2.inRange(img, lower_rgb_yellow, upper_rgb_yellow)

	# rgb thresholding for white
	lower_rgb_yellow = np.array([200, 100, 100])
	upper_rgb_yellow = np.array([255, 255, 255])
	mask_rgb_white = cv2.inRange(img, lower_rgb_yellow, upper_rgb_yellow)

	# hls thresholding for yellow
	lower_hls_yellow = np.array([20, 120,  80])
	upper_hls_yellow = np.array([45, 200, 255])
	mask_yellow = cv2.inRange(hls, lower_hls_yellow, upper_hls_yellow)

	color_binary = np.zeros_like(s_channel)
	color_binary[(mask_rgb_yellow == 255)|(mask_rgb_white==255)|(mask_yellow==255)]= 1

	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

	# Combining threshold 
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(color_binary==1) | (sxbinary==1)] = 1

	return combined_binary

def get_warp_points():
	corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
	new_top_left  = np.array([corners[0, 0], 0])
	new_top_right = np.array([corners[3, 0], 0])
	offset = [50, 0]
	src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
	dst_points = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])
	return src_points, dst_points

def perspective_transform(img, src, dst):
	image_size = (img.shape[1], img.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)
	M_inv = cv2.getPerspectiveTransform(dst, src)
	return warped, M, M_inv

def calc_histogram(img):
	return np.sum(img[img.shape[0]/2:,:], axis=0)

def find_lane_pixels(binary_warped):
	# Take a histogram of the bottom half of the image
	histogram = calc_histogram(binary_warped)

	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nwindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Set height of windows - based on nwindows above and image shape
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
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
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
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
		(nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

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

	return leftx, lefty, rightx, righty, out_img


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

def fit_polynomial(binary_warped):
	# Find our lane pixels first
	leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

	# Fit a second order polynomial to each using `np.polyfit`
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	try:
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	except TypeError:
		# Avoids an error if `left` and `right_fit` are still none or incorrect
		print('The function failed to fit a line!')
		left_fitx = 1*ploty**2 + 1*ploty
		right_fitx = 1*ploty**2 + 1*ploty

	left_curverad, right_curverad, lane_deviation = get_curvature(leftx, lefty, rightx, righty, ploty, binary_warped.shape)

	## Visualization
	out_img[lefty, leftx]   = [255, 0, 0]
	out_img[righty, rightx] = [0, 0, 255]

	# Plots the left and right polynomials on the lane lines
	for (x, y) in zip(left_fitx, ploty):
		cv2.circle(out_img,(int(x), int(y)), 1, (0, 255, 255), -1)
	for (x, y) in zip(right_fitx, ploty):
		cv2.circle(out_img,(int(x), int(y)), 1, (0, 255, 255), -1)

	return left_fitx, right_fitx, ploty, left_fit, right_fit, left_curverad, right_curverad, lane_deviation, out_img

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

def sanity_check(left_fitx, right_fitx, ploty):
	thresh = 100
	pts_left = np.vstack([left_fitx, ploty])
	pts_right = np.vstack([right_fitx, ploty])

	# check difference of lanes distanc
	diff1 = abs((pts_right[0][0] - pts_left[0][0]) - (pts_right[0][359] - pts_left[0][359]))
	diff2 = abs((pts_right[0][359] - pts_left[0][359]) - (pts_right[0][719] - pts_left[0][719]))
	return (diff1 < thresh and diff2 < thresh)

prev_left_fitx = []
prev_right_fitx = []
prev_left_curvature = 0.0
prev_right_curvature = 0.0
prev_lane_deviation = 0.0

def pipeline(img, mtx, dist):
	undistorted = undistort(img, mtx, dist)
	binary = combine_threshold(undistorted)
	src_points, dst_points = get_warp_points()
	warped_binary, M, M_inv = perspective_transform(binary, src_points, dst_points)
	left_fitx, right_fitx, ploty, left_fit, right_fit, left_curvature, right_curvature, lane_deviation, out_img = fit_polynomial(warped_binary)

	is_good = sanity_check(left_fitx, right_fitx, ploty)

	global prev_left_fitx, prev_right_fitx
	global prev_left_curvature, prev_right_curvature, prev_lane_deviation
	if is_good:
		prev_left_fitx = left_fitx
		prev_right_fitx = right_fitx
		prev_left_curvature = left_curvature
		prev_right_curvature = right_curvature
		prev_lane_deviation = lane_deviation
	else:
		# use previous result
		left_fitx = prev_left_fitx
		right_fitx = prev_right_fitx
		left_curvature = prev_left_curvature
		right_curvature = prev_right_curvature
		lane_deviation = prev_lane_deviation

	lanes = draw_lanes(undistorted, M_inv, left_fitx, right_fitx, ploty, left_curvature, right_curvature,lane_deviation)
	cv2.imshow('lane detection', lanes)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', help='Path to image or video', type=str)
	parser.add_argument('--dir', help='Directory of images', type=str)
	args = parser.parse_args()
	input_file_name = args.input
	directory_name  = args.dir

	param_file = 'camera_dist_pickle.p'
	mtx, dist = load_camera_param(param_file)

	if input_file_name:
		cap = cv2.VideoCapture(input_file_name)
		while True:
			ret, img = cap.read()
			if not ret:
				cv2.waitKey(0)
				break

			pipeline(img, mtx, dist)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	elif directory_name:
		filelist = glob.glob(directory_name + '/*.jpg')
		for filename in filelist:
			img = cv2.imread(filename)
			if img is None:
				break

			pipeline(img, mtx, dist)
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break

	if input_file_name:
		cap.release()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
