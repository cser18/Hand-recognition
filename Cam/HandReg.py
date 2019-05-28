import cv2
#添加模块和矩阵模块
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
    # 打开摄像头，若打开本地视频，同opencv一样，只需将０换成("×××.avi")
    # cap.set(cv2.CV_CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FPS, 30)

def init():
    ret, frame = cap.read()  # show a frame
    [H, W, C] = frame.shape
    frame = cv2.flip(frame, 1, dst=None)  # 水平镜像
    frame = cv2.rectangle(frame, (W - 250, H - 80), (W - 150, H - 200), (255, 0, 0), 2)
    frame_data = np.array(frame)
    [fH, fW, fN] = frame_data.shape
    frame_data = frame_data[H - 200: H - 80, W - 250:W - 150]
    return ret,frame_data


def skinRgb(frame,mark = None):
    R = 2
    G = 1
    B = 0
    [H,W,N] = frame.shape
    rgbRes = np.zeros((H,W,N), np.uint8)
    for h  in range(H):
        for w in range(W):
            if((frame[h][w][R] > 95 and frame[h][w][B] > 20 and int(frame[h][w][R]) - int(frame[h][w][B]) > 15 and int(frame[h][w][R]) - int(frame[h][w][G]) > 15)
                    or (frame[h][w][R] > 200 and frame[h][w][G] > 210 and frame[h][w][B] > 170 and abs(int(frame[h][w][R]) - int(frame[h][w][B])) <= 15 and frame[h][w][R] > frame[h][w][B] and frame[h][w][G] > frame[h][w][B])):
                rgbRes[h][w] = frame[h][w]
    for h  in range(H):
        for w in range(W):
            if rgbRes[h][w][0] != 0:
                rgbRes[h][w] = 255
    return rgbRes

def skinCrCb(frame):
    tem = frame.copy
    frame_crcb = cv2.cvtColor(frame,cv2.COLOR_RGB2YCrCb)
    skinCrCbHist = np.zeros((256,256),np.uint8)
    cv2.ellipse(skinCrCbHist, (int(113),int(155.6)),(int(23.4),int(15.2)), 43.0,0.0,360.0,(255,255,255), -1)
    #cv2.imshow('hist',skinCrCbHist)
    output_mask = np.zeros((120,100))
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if(skinCrCbHist[frame_crcb[i][j][2]][frame_crcb[i][j][1]] > 0):
                output_mask[i][j] = 255
    return output_mask


def skinHsv(rgbRes):
    temImg = rgbRes.copy()
    frame_data = cv2.cvtColor(rgbRes, cv2.COLOR_BGR2HSV_FULL)
    [H, S, V] = cv2.split(frame_data)
    H = np.array(H)
    S = np.array(S)  # 饱和度
    V = np.array(V)  # 明度
    [th, tw, tc] = frame_data.shape
    for i in range(th):
        for j in range(tw):
            hVal = H[i, j]
            sVal = S[i, j]
            vVal = V[i, j]
            if hVal <= 20 and hVal >= 7 and sVal >= 48 and vVal >= 50:
                H[i, j] = 255
            else:
                H[i, j] = 0
    for i in range(th):
        for j in range(tw):
            if H[i,j] == 0:
                temImg[i][j] = 0
    return temImg

# 直方图计算


def calcHist(frame):
    temImag = frame.copy()
    srcGray = cv2.cvtColor(temImag,cv2.COLOR_RGB2GRAY)
    #print(srcGray.shape) # (120 100)
    # for i in range(srcGray.shape[0]):
    #     for j in range(srcGray.shape[1]):
    #         if(srcGray[i][j] < 50 or srcGray[i][j] > 100):
    #             srcGray[i][j] = 255

    cv2.imshow("gray", srcGray)
    cv2.resizeWindow("gray", 200, 200)
    #hist = cv2.calcHist(srcGray, [0], None, [256],[0,256])

    #plt.hist(srcGray.ravel(),256,[0,256])
   # plt.show()
    return srcGray


def main():
    ret = True
    while(ret):
        ret, frame = init()
        tem = skinCrCb(frame)
        # B = 80
        # G = 70
        # R = 90
        # for i in range(frame.shape[0]):
        #     for j in range(frame.shape[1]):
        #         if(frame[i][j][0] < B):
        #             frame[i][j][0] = B
        #         if (frame[i][j][1] < G):
        #             frame[i][j][1] = G
        #         if (frame[i][j][2] < R):
        #             frame[i][j][2] = R
        # tem = skinCrCb(frame)
        #mark = calcHist(frame)
        #tem = skinHsv(frame)

        #rgbHand = skinRgb(tem)

        cv2.imshow("tem", tem)
        cv2.imshow("target", frame)
        #cv2.imshow("rgbHand",rgbHand)
        #cv2.imshow("hsvImg", hsvImg)
        #cv2.resizeWindow("hsvImg", 200, 200)
        cv2.resizeWindow("target", 200, 200)
        cv2.resizeWindow("rgbHand", 200, 200)
        cv2.resizeWindow("tem", 200, 200)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # 释放并销毁窗口
if  __name__ in '__main__':
    main()
