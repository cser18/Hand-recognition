import cv2

#添加模块和矩阵模块
import numpy as np


cap=cv2.VideoCapture(0)
#创建BackgroundSubtractorMOG2
fgbg = cv2.createBackgroundSubtractorMOG2()
# 打开摄像头，若打开本地视频，同opencv一样，只需将０换成("×××.avi")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 定义结构元素   椭圆结构
# cap.set(cv2.CV_CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FPS,30)
# cnt = 1
while(1):    # get a frame
    ret, frame = cap.read()    # show a fram

    [H,W,C] = frame.shape
    frame = cv2.flip(frame, 1, dst=None)  # 水平镜像
    frame = cv2.rectangle(frame,(W - 250,H - 80),(W - 150,H - 200),(255, 0, 0), 2)
    frame_data = np.array(frame)
    [fH,fW,fN] = frame_data.shape
    frame_data = frame_data[H - 200 : H - 80, W - 250:W - 150]
    ### 博客上的对图像处理的步骤
    minValue = 70
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # 阈值处理
    ret, bRes = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ###########################


    tframe = frame_data
    # 加入高斯背景消除法
    frame_mark = fgbg.apply(frame_data)
    # frame_data = cv2.medianBlur(frame_data, 3) 中值滤波
    frame_data = cv2.bilateralFilter(frame_data,5,15,15) #双边滤波
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2HSV_FULL)
    [H,S,V] = cv2.split(frame_data)
    # frame_data = cv2.morphologyEx(frame_data, cv2.MORPH_OPEN, kernel)
    # 转化为二值图
    # ret, frame_data = cv2.threshold(frame_data, 100, 255, cv2.THRESH_BINARY_INV)
    [th,tw,tc] = frame_data.shape
    H = np.array(H)
    S = np.array(S)  # 饱和度
    V = np.array(V)  # 明度
    for i  in range(th):
        for j in range(tw):
            hVal = H[i,j]
            sVal = S[i,j]
            vVal = V[i,j]
            if  hVal <= 35 and hVal >= 0 and sVal > 48 and vVal > 50:
                H[i,j] = 255
            else:
                H[i,j] = 0

    # 将H通道进行转换#########################
    # 发现 正要H通道获得的信息槽点小 所获得的图片的效果可以作为深度学习的喂入图片
    minValue = 70
    # hgray = cv2.cvtColor(H, cv2.COLOR_BGR2GRAY)
    hblur = cv2.GaussianBlur(H, (5, 5), 2)
    hth3 = cv2.adaptiveThreshold(hblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # 阈值处理
    ret, hbRes = cv2.threshold(hth3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ###########################################################
    res = cv2.morphologyEx(H, cv2.MORPH_OPEN, kernel)     # 形态学开运算去噪点
    im, contours, hierarchy = cv2.findContours(H, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 300:
            [x,y,w,h] = cv2.boundingRect(c)      #找到这个矩形 画下来
            # 画矩形
            cv2.rectangle(tframe,(x,y),(x+w,y+h),(0,255,0), 2)


    cv2.imshow("capture", frame)
    cv2.imshow("target",frame_data)
    cv2.imshow("H", H)
    cv2.imshow("res",res)
    cv2.imshow("gaosi",frame_mark)
    cv2.imshow("getTarget",tframe)
    cv2.imshow("boke", bRes)
    cv2.imshow("hboke",hbRes)
    cv2.resizeWindow("H", 200, 200)
    cv2.resizeWindow("target", 200, 200)
    cv2.resizeWindow("res",200,200)
    cv2.resizeWindow("gaosi",200,200)
    cv2.resizeWindow("getTarget",200,200)
    cv2.resizeWindow("boke",200, 200)
    cv2.resizeWindow("hboke",200,200)
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("1.jpg",H)
cap.release()
cv2.destroyAllWindows()
#释放并销毁窗口
