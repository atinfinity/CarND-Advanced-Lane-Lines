import numpy as np
import cv2
import pickle
import glob

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
PATTERN_SIZE = (9, 6)

def calc_reprojection_error(imgpoints, objpoints, mtx, dist, rvecs, tvecs):
	mean_error = 0
	for i in range(len(objpoints)):
		imgpoints_, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i], imgpoints_, cv2.NORM_L2) / len(imgpoints_)
		mean_error += error
	return mean_error

def main():
	objpoints = []
	imgpoints = []

	objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32) 
	objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

	filelist = glob.glob('camera_cal/*.jpg')
	for filename in filelist:
		img  = cv2.imread(filename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

		if ret == True:
			corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
			imgpoints.append(corners_subpix)
			objpoints.append(objp)

	# camera calibration
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	dist_pickle = {}
	dist_pickle['mtx'] = mtx
	dist_pickle['dist'] = dist
	pickle.dump(dist_pickle, open('camera_dist_pickle.p', 'wb'))

	# calc re-projection error
	mean_error = calc_reprojection_error(imgpoints, objpoints, mtx, dist, rvecs, tvecs)
	print('re-projection error: {}'.format(mean_error / len(objpoints)))


if __name__ == '__main__':
	main()

