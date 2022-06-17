# Imports:
import cv2
import math
import numpy as np

# def rotateBound(image, angle):
#     # grab the dimensions of the image and then determine the
#     # center
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#     # perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (nW, nH))                     

# image path
# path = "E://Nikhil//HP Dataset//OpenCV//"
# fileName = "Left_sign.jpg"

cap=cv2.VideoCapture(0)
while True:
    ret,inputImage=cap.read()

    # Reading an image in default mode:
    # inputImage = cv2.imread(path + fileName)

    # Grayscale conversion:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    grayscaleImage = 255 - grayscaleImage

    # Find the big contours/blobs on the binary image:
    contours, hierarchy = cv2.findContours(grayscaleImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour 1-1:
    for i, c in enumerate(contours):

        # Approximate the contour to a polygon:
        contoursPoly = cv2.approxPolyDP(c, 3, True)

        # Convert the polygon to a bounding rectangle:
        boundRect = cv2.boundingRect(contoursPoly)

        # Get the bounding rect's data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Get the rect's area:
        rectArea = rectWidth * rectHeight

        minBlobArea = 100
        # Check if blob is above min area:
        if rectArea > minBlobArea:
            
            # Crop the roi:
            croppedImg = grayscaleImage[rectY:rectY + rectHeight, rectX:rectX + rectWidth]

            # Extend the borders for the skeleton:
            borderSize = 5        
            croppedImg = cv2.copyMakeBorder(croppedImg, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_CONSTANT)

            # Store a deep copy of the crop for results:
            grayscaleImageCopy = cv2.cvtColor(croppedImg, cv2.COLOR_GRAY2BGR)

            # Compute the skeleton:
            skeleton = cv2.ximgproc.thinning(croppedImg, None, 1)

            # Threshold the image so that white pixels get a value of 0 and
            # black pixels a value of 10:
            _, binaryImage = cv2.threshold(skeleton, 128, 10, cv2.THRESH_BINARY)

            # Set the end-points kernel:
            h = np.array([[1, 1, 1],
                        [1, 10, 1],
                        [1, 1, 1]])

            # Convolve the image with the kernel:
            imgFiltered = cv2.filter2D(binaryImage, -1, h)

            # Extract only the end-points pixels, those with
            # an intensity value of 110:
            binaryImage = np.where(imgFiltered == 110, 255, 0)
            # The above operation converted the image to 32-bit float,
            # convert back to 8-bit uint
            binaryImage = binaryImage.astype(np.uint8)

            # Find the X, Y location of all the end-points
            # pixels:
            Y, X = binaryImage.nonzero()

            # Check if I got points on my arrays:
            if len(X) > 0 or len(Y) > 0:

                # Reshape the arrays for K-means
                Y = Y.reshape(-1,1)
                X = X.reshape(-1,1)
                Z = np.hstack((X, Y))
                # print("Z:",Z)
                # K-means operates on 32-bit float data:
                floatPoints = np.float32(Z)
                # print("floatpoints:",floatPoints)

                # Set the convergence criteria and call K-means:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, label, center = cv2.kmeans(floatPoints, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                # print("Center:",center)

                # Set the cluster count, find the points belonging
                # to cluster 0 and cluster 1:
                cluster1Count = np.count_nonzero(label)
                cluster0Count = np.shape(label)[0] - cluster1Count
                # Look for the cluster of max number of points
                # That cluster will be the tip of the arrow:
                maxCluster = 0
                if cluster1Count > cluster0Count:
                    maxCluster = 1

                # Check out the centers of each cluster:
                matRows, matCols = center.shape
                # We need at least 2 points for this operation:
                if matCols >= 2:
                    # Store the ordered end-points here:
                    orderedPoints = [None] * 2
                    # Let's identify and draw the two end-points
                    # of the arrow:
                    for b in range(matRows):
                        # print("b:",b)
                        # Get cluster center:
                        pointX = int(center[b][0])
                        pointY = int(center[b][1])
                        # Get the "tip"
                        if b == maxCluster:
                            color = (0, 0, 255)
                            orderedPoints[0] = (pointX, pointY)
                        # Get the "tail"
                        else:
                            color = (255, 0, 0)
                            orderedPoints[1] = (pointX, pointY)
                            # print(orderedPoints)
                    # Draw it:
                    cv2.circle(grayscaleImageCopy,orderedPoints[0] , 3,(0, 0, 255), -1)
                    cv2.circle(grayscaleImageCopy,orderedPoints[1] , 3,(255, 0, 0), -1)
                    cv2.imshow("End Points", grayscaleImageCopy)
                    key=cv2.waitKey(1)
                    if key==ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        break            
                    
                    # Store the tip and tail points:
                    # p0x = orderedPoints[1][0]
                    # p0y = orderedPoints[1][1]
                    # p1x = orderedPoints[0][0]
                    # p1y = orderedPoints[0][1]
                    # Compute the sides of the triangle:
                    # adjacentSide = p1x - p0x
                    # oppositeSide = p0y - p1y
                    # # Compute the angle alpha:
                    # alpha = math.degrees(math.atan(oppositeSide / adjacentSide))
                    # # Adjust angle to be in [0,360]:
                    # if adjacentSide < 0 < oppositeSide:
                    #     alpha = 180 + alpha
                    # else:
                    #     if adjacentSide < 0 and oppositeSide < 0:
                    #         alpha = 270 + alpha
                    #     else:
                    #         if adjacentSide > 0 > oppositeSide:
                    #             alpha = 360 + alpha



                    # # Deep copy for rotation (if needed):
                    # rotatedImg = croppedImg.copy()
                    # # Undo rotation while padding output image:
                    # rotatedImg = rotateBound(rotatedImg, alpha)
                    # cv2. imshow("rotatedImg", rotatedImg)
                    # cv2.waitKey(0)

                else:
                    print( "K-Means did not return enough points, skipping..." )
            else:
                print( "Did not find enough end points on image, skipping..." )           



