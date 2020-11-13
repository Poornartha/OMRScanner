from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

ANSWER_KEY = {0: 1, 1: 2, 2: 0, 3: 2, 4: 1, 5: 1, 6: 2, 7: 3, 8: 1, 9: 0}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Detecting the Paper
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None
sections = []
secCount = 0

if len(cnts) > 0:
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		
		# 4 Vertices of a Paper:
		if len(approx) == 4:
			docCnt = approx
			sections.append(docCnt)
			secCount += 1
			if secCount == 3:
				break

# Prespective Transformation:
paper1 = four_point_transform(image, sections[0].reshape(4, 2))
warped1 = four_point_transform(gray, sections[0].reshape(4, 2))
paper2 = four_point_transform(image, sections[1].reshape(4, 2))
warped2 = four_point_transform(gray, sections[1].reshape(4, 2))
paper3 = four_point_transform(image, sections[2].reshape(4, 2))
warped3 = four_point_transform(gray, sections[2].reshape(4, 2))

# Thresholding => Differing Foreground from Background:
thresh1 = cv2.threshold(warped1, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

thresh2 = cv2.threshold(warped2, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

thresh3 = cv2.threshold(warped3, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# cv2.imshow("Section1", thresh1)
# cv2.imshow("Section2", thresh2)
# cv2.imshow("Section3", thresh3)
# cv2.waitKey(0)


# Finding Bubbles using Contours:
cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []


for c in cnts:
	# Finding Contours with Aspect Ratio 1 => Will help us to find bubbles
	
	# Finding Bounding Boxes:
	(x, y, w, h) = cv2.boundingRect(c)

	# Finding Aspect Ratio:
	ar = w / float(h)

	# Validating Width / Height of the Bubbles and an Aspect Ratio of 1:
	# Change: Assumption => Width and Height are assumed.
	if ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

# Grading the Bubbles Found:
# Change: Assumptions =>
# 1. Every Row has 5 Bubbles 
# 2. Questions are to Answered Top-To-Bottom

print(len(questionCnts))
questionCnts = contours.sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0

# Searching Contours at every row:
for (q, i) in enumerate(np.arange(0, len(questionCnts), 1)):
	cnts = contours.sort_contours(questionCnts[i:i + 4], method="left-to-right")[0]
	print(cnts)
	print("-------")
	bubbled = None
	print(i, i+4)
	# Looping over the contours within the row => cnts is the contours present in a given row:
	for (j, c) in enumerate(cnts):

		# Mask for the current Bubble:
		mask = np.zeros(thresh2.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		# Apply the mask:
		mask = cv2.bitwise_and(thresh2, thresh2, mask=mask)
		total = cv2.countNonZero(mask)
		# print("Bubbled", bubbled)
		# print("Total", total)

		# If total None Zero Pixels is > One of the bubbled circles:
		# Simple Maximization Technique:
		# Change: For Multiple Colored Circles use a treshold value.

		# if bubbled is None or total > bubbled[0]:
		
		bubbled = (total, j)

	# Color Coding for incorrect Answers => Red
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# Check if the bubbled answer is correct => Green:
	if total == bubbled[1]:
		# Color Coding for correct Answers
		color = (0, 255, 0)
		correct += 1

	cv2.drawContours(paper2, [cnts[k]], -1, color, 3)
	cv2.imshow("Original", paper2)
	cv2.waitKey(0)

"""	
	cv2.drawContours(paper2, [cnts[k]], -1, color, 3)

# Printing the Score:
# Grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper2, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)

"""	










