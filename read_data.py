import cv2

def get_queue():
    images=[]
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    # initialize webcam capture object
    cap = cv2.VideoCapture('test.mp4')
    # 1920, 1080
    # cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    frames = 0
    while(True):
        ret,frame = cap.read()
        if not ret:
            break
        # if frames % 12==0:
        dim = (1920, 1080)
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        images.append(img)
        frames +=1
        
        if frames==328:#328
            print(len(images))
            cv2.destroyAllWindows()
            cap.release()
            print('end capture images successfully')
            return images
        
# get_queue()