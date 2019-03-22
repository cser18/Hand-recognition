import cv2

import numpy as np#添加模块和矩阵模块

cap=cv2.VideoCapture(0)
# 打开摄像头，若打开本地视频，同opencv一样，只需将０换成("×××.avi")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
# cap.set(cv2.CV_CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FPS,30)
while(1):    # get a frame   
    ret, frame = cap.read()    # show a frame
    [H,W,C] = frame.shape
    frame = cv2.flip(frame, 1, dst=None)  # 水平镜像
    frame = cv2.rectangle(frame,(W - 250,H - 80),(W - 150,H - 200),(255, 0, 0), 2)
    frame_data = np.array(frame)
    frame_data = frame_data[H - 200 : H - 80, W - 250:W - 150]
    # frame_data = cv2.blur(frame_data,(2,2))
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2HSV_FULL)
    [H,S,V] = cv2.split(frame_data)
    # frame_data = cv2.morphologyEx(frame_data, cv2.MORPH_OPEN, kernel)
    # 转化为二值图
    # ret, frame_data = cv2.threshold(frame_data, 100, 255, cv2.THRESH_BINARY_INV)
    [th,tw,tc] = frame_data.shape
    H = np.array(H)
    S = np.array(S)
    V = np.array(V)
    for i  in range(th):
        for j in range(tw):
            hVal = H[i,j]
            sVal = S[i,j]
            vVal = V[i,j]
            if  hVal <= 35 and hVal >= 0 and sVal > 48 and vVal > 50:
                H[i,j] = 255
            else:
                H[i,j] = 0
    dit = cv2.dilate(H,kernel)
    dit = cv2.dilate(dit,kernel)
    erode = cv2.erode(H,kernel)
    erode = cv2.erode(erode,kernel)
    res = dit - erode
    cv2.imshow("capture", frame)
    cv2.imshow("target",frame_data)
    cv2.imshow("H", H)
    cv2.imshow("res",res)
    cv2.resizeWindow("H", 200, 200)
    cv2.resizeWindow("target", 200, 200)
    cv2.resizeWindow("res",200,200)
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("1.jpg",H)

cap.release()
cv2.destroyAllWindows()
#释放并销毁窗口
