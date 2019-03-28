import cv2
#添加模块和矩阵模块
import numpy as np

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


def skinRgb(frame):
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
    return rgbRes

def skinHsv(rgbRes):
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
            if hVal <= 29 and hVal >= 7 and sVal > 46 and vVal > 50:
                H[i, j] = 255
            else:
                H[i, j] = 0
    return H

def main():
    ret = True
    while(ret):
        ret, frame = init()
        rgbHand = skinRgb(frame)
        hsvImg = skinHsv(rgbHand)
        cv2.imshow("target", frame)
        cv2.imshow("rgbHand",rgbHand)
        cv2.imshow("hsvImg", hsvImg)
        cv2.resizeWindow("hsvImg", 200, 200)
        cv2.resizeWindow("target", 200, 200)
        cv2.resizeWindow("rgbHand", 200, 200)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # 释放并销毁窗口
if  __name__ in '__main__':
    main()
