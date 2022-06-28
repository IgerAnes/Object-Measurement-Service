import pickle
import socket
import cv2
import threading
from datetime import datetime
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from scipy.spatial import distance as dist
import argparse


# setup server parameter
HOST = "0.0.0.0"
PORT = 6000
HEADERSIZE = 64
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

# setup opencv parameter
PIXELSPERMETRIC = None

def systemTime():
    now = datetime.now()
    timeFormat = "%Y-%m-%d %H:%M:%S"
    formatTime = datetime.strftime(now, timeFormat)
    return formatTime

# setup socket server with parameter
socketServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    socketServer.bind((HOST, PORT))
except socket.error as e:
    print(f"[{systemTime()}][ERROR] {str(e)}")

def handleClient(conn, addr):
    print(f"[{systemTime()}][NEW CONNECTION] New user connects from {addr[0]}:{addr[1]}")
    connectedFlag = True
    imageCounter = 0
    while connectedFlag:
        data = b""
        data_len = conn.recv(HEADERSIZE).decode(FORMAT)
        # print(f"[{systemTime()}][SERVER] Receive data length")
        if data_len != None and data_len != "":
            data_len = int(data_len)
            remain_len = data_len
            print(f"[{systemTime()}][{addr[0]}:{addr[1]}] data length: {data_len}")
            # check remain data length, and avoid to get data from the other frame
            while len(data) < data_len:
                if remain_len >= 4096:
                    data += conn.recv(4096)
                    remain_len -= 4096
                else:
                    data += conn.recv(remain_len)
            
            # convert byte data to python object by pickle
            origin_data = pickle.loads(data[:data_len])
            if isinstance(origin_data, str):
                print(f"[{systemTime()}][SERVER] Check origin data type")
                if origin_data == DISCONNECT_MESSAGE:
                    connectedFlag = False
                    print(f"[{systemTime()}][SERVER] Disconnect from client {addr[0]}:{addr[1]}")
            else:
                imageCounter += 1
                # add your image process function here
                
                # cv2.putText(origin_data, f"Counter: {imageCounter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                #         1, (255, 100, 50), 1 , cv2.LINE_AA)
                # print(f"[{systemTime()}][SERVER] OpenCV process counter: {imageCounter}")
                # cv2.imwrite('currentFrame.png', origin_data)
                
                # Turn image to gray and do gaussian blur
                grayImage = cv2.cvtColor(origin_data, cv2.COLOR_BGR2GRAY)
                grayImageGaussian = cv2.GaussianBlur(grayImage, (7, 7), 0)
                
                # Run canny edge detection find all object edge, 
                # and use dilation and erosion to close gap between object edge 
                edgeDetection = cv2.Canny(grayImageGaussian, 50, 100)
                edgeDetection = cv2.dilate(edgeDetection, None, iterations=1)
                edgeDetection = cv2.erode(edgeDetection,  None, iterations=1)
                
                # Find contour in edge map
                contourFinder = cv2.findContours(edgeDetection.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contourFinder = imutils.grab_contours(contourFinder)
                
                # Sort the contours from left-to-right and initilaize the 'pixils per metric' calibration value
                (contourFinder, _) = contours.sort_contours(contourFinder)
                
                for c in contourFinder:
                    # check if contour is large enough, otherwise ignore it
                    if cv2.contourArea(c) < 100:
                        continue
                    # Compute the rotated bounding box of contour
                    origImage = origin_data.copy()
                    boxData = cv2.minAreaRect()
                    boxData = cv2.cv.BoxPoints(boxData) if imutils.is_cv2() else cv2.boxPoints(boxData)
                    boxData = np.array(boxData, dtype="int")
                    
                    # Order the points in the contour such that thet appear in
                    # top-left, top-right, bottom-right, and bottom-left
                    # order, then draw the outlier of rotated bounding box
                    boxData = perspective.order_points(boxData)
                    cv2.drawContours(origImage, [boxData.astype("int")], -1, (0, 255, 0), 2)
                    
                    # loop over the original points adn draw them
                    for (x, y) in boxData:
                        cv2.circle(origImage, (int(x), int(y)), 5, (0, 0, 255), -1)
                    
                    # Unpack ordered bounding box, then compute the midpoint, and draw it
                    def midpoint(pointA, pointB):
                        return ((pointA[0]+pointB[0])*0.5, (pointA[1]+pointB[1])*0.5)
                    
                    (topleft, topright, bottomright, bottomleft) = boxData
                    (centerX, centerY) = midpoint(topleft, bottomright)
                    (tltrX, tltrY) = midpoint(topleft, topright)
                    (tlblX, tlblY) = midpoint(topleft, bottomleft)
                    (trbrX, trbrY) = midpoint(topright, bottomright)
                    (blbrX, blbrY) = midpoint(bottomleft, bottomright)
                    
                    cv2.circle(origImage, (int(centerX), int(centerY)), 5, (255, 0, 0), -1)
                    
                    # Measuring the object size by bounding box midpoint, compute the euclidean distance
                    distanceA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    distanceB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    
                    # compute the size of the object
                    dimA = distanceA / PIXELSPERMETRIC
                    dimB = distanceB / PIXELSPERMETRIC
                    
                    # Draw object-size on image
                    cv2.putText(origImage, "{:.4f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                    cv2.putText(origImage, "{:.4f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                    
                    cv2.imwrite("measure-result-image.png", origImage)
                    
        
        
    print(f"[{systemTime()}][SERVER] Start to close client {addr[0]}:{addr[1]} connection")
    conn.close()
    print(f"[{systemTime()}][SERVER] Client {addr[0]}:{addr[1]} connection closed")


def serverStart():
    socketServer.listen()
    print(f"[{systemTime()}][SERVER] Server Address: {HOST}:{PORT}")
    print(f"[{systemTime()}][SERVER] Start to listen to client connection ...")
    while True:
        # accept the connectio form outside
        print("Start to accept connection")
        conn, addr = socketServer.accept() # this cmd. will wait until receive next connection
        thread = threading.Thread(target=handleClient, args=(conn, addr))
        thread.start()
        print(f"[{systemTime()}][ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

print(f"[{systemTime()}][SERVER] Server is starting ...")
serverStart()