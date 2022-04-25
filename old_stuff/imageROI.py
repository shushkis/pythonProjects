import cv2


ref_point = []
cropping = False
ever_cropped = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, ever_cropped

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

    # draw a rectangle on top of the image
    if (len(ref_point) > 1):
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

        roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]  # crop that rect to a new image
        if (ref_point[0][1]-ref_point[1][1] != 0 and ref_point[0][0]-ref_point[1][0] != 0): #check sanity of window width and hight
            ref_point = []
            ever_cropped = True # so I know that in some point a new image win was open
            cv2.imshow("ROI", roi) # draw copped
            cv2.waitKey(0)
        
def rotate(degree):
    degree = cv2.getTrackbarPos('degree','image')
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1) # get R/T in 2D for deg
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) # apply affine transformation
    cv2.imshow('image', rotated_image) # show rotated
    
image = cv2.imread("brain.jpeg")
clone = image.copy() # make a copy
cv2.namedWindow("image")


degree = 0
height, width = image.shape[:2]
cv2.createTrackbar('degree','image',degree,360,rotate)
cv2.setMouseCallback("image", click_and_crop)



def clear():
    global ref_point
    image = cv2.imread("brain.jpeg")
    cv2.imshow('image',image)
    ref_point = []

cv2.imshow('image',image)

while True:    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # esc key
            break

    if key == ord("c"):
        clear()
        
    if (cv2.getWindowProperty('ROI', 0) < 0 and ever_cropped):
        ever_cropped = False
        clear()

        

cv2.destroyWindow('image')           
        




