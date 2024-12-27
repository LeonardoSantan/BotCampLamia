import cv2
import numpy as np

image = cv2.imread('./Card - 21/image.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

blue_lower = np.array([100, 150, 50])
blue_upper = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
blue_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, blue_kernel)
blue_result = cv2.bitwise_and(image, image, mask=blue_mask)

red_lower1 = np.array([0, 150, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 150, 50])
red_upper2 = np.array([180, 255, 255])
red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
red_mask = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)
red_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_kernel)
red_result = cv2.bitwise_and(image, image, mask=red_mask)

green_lower = np.array([40, 100, 50])
green_upper = np.array([80, 255, 255])
green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
green_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, green_kernel)
green_result = cv2.bitwise_and(image, image, mask=green_mask)

cv2.imshow('Original Image', image)
cv2.imshow('Blue Mask', blue_mask)
cv2.imshow('Blue Detection', blue_result)
cv2.imshow('Red Mask', red_mask)
cv2.imshow('Red Detection', red_result)
cv2.imshow('Green Mask', green_mask)
cv2.imshow('Green Detection', green_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
