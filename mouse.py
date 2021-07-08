import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy


##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
right_click_fingers = [1,0,0,0,0]
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img,draw=False)
    # 2. Get key landmarks location
    if len(lmList) != 0:
        x0, y0 = lmList[0][1:]
        x4, y4 = lmList[4][1:]
        x5 ,y5 = lmList[5][1:]
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, 100), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 2)
        # 4. Convert Coordinates
        x3 = np.interp(x0, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y0, (250, hCam -frameR), (0, hScr))
        
        # 5. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        
        # 6. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x0, y0), 15, (255, 0, 0), cv2.FILLED)
        plocX, plocY = clocX, clocY
            
        # 7. clicking mode
        if fingers[0] == 1 :
            # 8 . Find distance between landmarks [4] and [5]
            length, img, lineInfo = detector.findDistance(4, 5, img,draw=False)
            # 9. Left click mouse if distance short
            if length < 40:
                autopy.mouse.click()
                print("Left Click")
                time.sleep(0.1)
            # 10. Right click 
            elif fingers == right_click_fingers:  # right_click_fingers = [1,0,0,0,0]
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
                print("Right Click")
                time.sleep(0.1)
            
    
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27 : 
        break 
cap.release()
cv2.destroyAllWindows()
