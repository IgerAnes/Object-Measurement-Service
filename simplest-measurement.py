import cv2
from datetime import datetime
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
from scipy.spatial import distance as dist
import time
import math
import os
import mvsdk


# setup opencv parameter
# PIXELSPERMETRIC = None
PIXELSPERMETRIC = 6.5549745
POINTX = np.array([876, 1034])
POINTY = np.array([738, 908])
ORIGINPOINT = np.array([744, 1038])
CALIBRATEVALUE = 5*PIXELSPERMETRIC
ROIAREA = np.array([[[744 + CALIBRATEVALUE, 1038 - CALIBRATEVALUE], [1920, 1002 - CALIBRATEVALUE], [1920, 0], [696 + CALIBRATEVALUE, 0]]], dtype=np.int32)

def systemTime():
    now = datetime.now()
    timeFormat = "%Y-%m-%d %H:%M:%S"
    formatTime = datetime.strftime(now, timeFormat)
    return formatTime


# DSHOW parameter is for windows only, if you use linux, change to CAP_V4L, CAP_V4L2, CAP_FFMPEG, CAP_GSTREAMER
# cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# if cam.isOpened():
#     print("[Camera] Succeed to get webcam data.")

counter = 0
current_time = 0
fps = 0
previous_time = 0
total_process_time = 0

# Industrial Camera Initialize
# 枚举相机
DevList = mvsdk.CameraEnumerateDevice()
nDev = len(DevList)
if nDev < 1:
    print("No camera was found!")

for i, DevInfo in enumerate(DevList):
    print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
i = 0 if nDev == 1 else int(input("Select camera: "))
DevInfo = DevList[i]
print(DevInfo)

# 打开相机
hCamera = 0
try:
    hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
except mvsdk.CameraException as e:
    print("CameraInit Failed({}): {}".format(e.error_code, e.message) )

# 获取相机特性描述
cap = mvsdk.CameraGetCapability(hCamera)
print(cap)

# 判断是黑白相机还是彩色相机
monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
if monoCamera:
    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
else:
    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

# 相机模式切换成连续采集
mvsdk.CameraSetTriggerMode(hCamera, 0)

# # 手动曝光，曝光时间30ms
# mvsdk.CameraSetAeState(hCamera, 0)
# mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

# 让SDK内部取图线程开始工作
mvsdk.CameraPlay(hCamera)

# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

# 分配RGB buffer，用来存放ISP输出的图像
# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

# setup camera height and width
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cam.set(cv2.CAP_PROP_FPS, 60)
# cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

try:
    while (cv2.waitKey(1) & 0xFF) != ord('q'):
		# 从相机取一帧图片
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

            frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_LINEAR)
            counter += 1
            frame = imutils.rotate(frame, 180)
            origin_data = frame
            
            # Turn image to gray and do gaussian blur
            grayImage = cv2.cvtColor(origin_data, cv2.COLOR_BGR2GRAY)
            grayImageGaussian = cv2.GaussianBlur(grayImage, (7, 7), 0)
            
            # Create a mask by ROI
            mask = np.zeros_like(grayImageGaussian)
            cv2.fillPoly(mask, ROIAREA, 255)
            grayImageGaussian = cv2.bitwise_and(grayImageGaussian, mask)
            # cv2.imwrite("maskImage.png", grayImageGaussian)
            

            # Run canny edge detection find all object edge, 
            # and use dilation and erosion to close gap between object edge 
            # edgeDetection = cv2.Canny(grayImage, 50, 100)
            edgeDetection = cv2.Canny(grayImageGaussian, 50, 200)
            edgeDetection = cv2.dilate(edgeDetection, None, iterations=1)
            edgeDetection = cv2.erode(edgeDetection,  None, iterations=1)
            # cv2.imshow('Edge', edgeDetection)
            
            # cv2.imwrite("canny-edge.png", edgeDetection)
            
            try:
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
                    boxData = cv2.minAreaRect(c)
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
                    # cv2.putText(origImage, "topleft", (int(topleft[0] + 15), int(topleft[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    # cv2.putText(origImage, "topright", (int(topright[0] + 15), int(topright[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    # cv2.putText(origImage, "bottomright", (int(bottomright[0] + 15), int(bottomright[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    # cv2.putText(origImage, "bottomleft", (int(bottomleft[0] + 15), int(bottomleft[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    
                    (centerX, centerY) = midpoint(topleft, bottomright)
                    (tltrX, tltrY) = midpoint(topleft, topright)
                    (tlblX, tlblY) = midpoint(topleft, bottomleft)
                    (trbrX, trbrY) = midpoint(topright, bottomright)
                    (blbrX, blbrY) = midpoint(bottomleft, bottomright)
                    
                    # Calculate rotate angle
                    MO2X = (POINTX[1] - ORIGINPOINT[1])/(POINTX[0] - ORIGINPOINT[0])
                    Mbl2br = (bottomright[1] - bottomleft[1])/(bottomright[0] - bottomleft[0])
                    tanMOXMbl2br = abs((Mbl2br - MO2X)/(1 + MO2X*Mbl2br))
                    rotateAngle = math.atan(tanMOXMbl2br)*180 / math.pi
                    cv2.putText(origImage, "Angle(Degree): {:.4f}".format(rotateAngle), (int(bottomleft[0]), int(bottomleft[1] + 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,51), 2)
                    
                    cv2.circle(origImage, (int(centerX), int(centerY)), 5, (255, 0, 0), -1)
                    
                    # Measuring the object size by bounding box midpoint, compute the euclidean distance
                    distanceA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    distanceB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    
                    # compute the size of the object
                    dimA = distanceA / PIXELSPERMETRIC
                    dimB = distanceB / PIXELSPERMETRIC
                    
                    # Draw object-size on image
                    cv2.putText(origImage, "{:.4f}mm".format(dimA), (int(tlblX - 15), int(tlblY - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                    cv2.putText(origImage, "{:.4f}mm".format(dimB), (int(tltrX + 10), int(tltrY)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
                    
                    # cv2.imwrite("measure-result-image.png", origImage)
                current_time = time.time()
                total_process_time = (current_time - previous_time)
                fps = 1 / total_process_time
                cv2.putText(origImage, f"Counter: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (50, 0, 255), 2 , cv2.LINE_AA)
                cv2.putText(origImage, f"FPS: {fps}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (50, 0, 255), 2 , cv2.LINE_AA)
                    
                cv2.imshow('Measurment-result', origImage)
            except:
                current_time = time.time()
                total_process_time = (current_time - previous_time)
                fps = 1 / total_process_time
                cv2.putText(frame, f"Counter: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (50, 0, 255), 2 , cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {fps}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (50, 0, 255), 2 , cv2.LINE_AA)
                cv2.imshow('Measurment-result', frame)
            
            print(f"Current Frame: {counter}")
            print(f"Current Process Time: {total_process_time*1000} ms")
            print(f"Current FPS: {fps}")

            previous_time = current_time

            cv2.imshow("Press q to end", frame)	
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
    
    # 关闭相机
    mvsdk.CameraUnInit(hCamera)

    # 释放帧缓存
    mvsdk.CameraAlignFree(pFrameBuffer)
    

except Exception as e:
    print(f"[Error]: {e}")

except KeyboardInterrupt:
    cv2.destroyAllWindows()




                    