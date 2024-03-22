import cv2
import numpy as np
import imutils
from collections import deque
 
# threshold untuk nilai oranye HSV
orangeLower = (0, 73, 198)
orangeUpper = (47, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

# List untuk titik tracking, pakai queue
pts = deque(maxlen=32)

# 5x5 matriks untuk kernel dilate erode
kernel = np.ones((5, 5), np.uint8)
 
# Akses video/kamera
cap = cv2.VideoCapture('Video Robot.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  	# Ambil per frame-nya
	ret, frame = cap.read()

	# resize frame, diblur, dan diubah ke HSV color space
	frame = imutils.resize(frame, width=600)
	# cv2.imshow('Original Image', frame)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# cv2.imshow('Blurred Image',blurred)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# cv2.imshow('HSV Image',hsv)

	# buat masking untuk warna oranye, dilate erode untuk masking yg lebih baik
	mask = cv2.inRange(hsv, orangeLower, orangeUpper)
	mask = cv2.erode(mask, kernel, iterations=2)
	mask = cv2.dilate(mask, kernel, iterations=2)
	# cv2.imshow('Threshold Image',mask)

	# Cari kontur dari mask dan titik pusat/center(x,y) dari bola
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# Kalau kontur tidak 0
	if len(cnts) > 0:
		# cari kontur terbesar di mask, lalu dipakai untuk menghitung minimum enclosing circle dan centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c) # Membuat lingkaran berdasarkan kontur yg terdeteksi
		M = cv2.moments(c) # pakai moments untuk cari area kontur
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# Apabila radius bola sesuai
		if radius > 10:
			# menggambar lingkaran pada bola dan titik pusat bola
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# update queue berisi titik tengah bola
	pts.appendleft(center)

	# loop menggunakan list titik tengah
	for i in range(1, len(pts)):
		# Apabila tidak ada tracked points
		if pts[i - 1] is None or pts[i] is None:
			continue
		# menggambar garis tracking
		thickness = int(np.sqrt(32 / float(i + 1)) * 1.0)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# Menampilkan hasil
	cv2.imshow('Frame',frame)
	key = cv2.waitKey(30)
	if (key == ord("x")):
		break
	if (key == ord("p")):
		cv2.waitKey(-1)

# Release video
cap.release()
 
# Tututp semua frame
cv2.destroyAllWindows()
