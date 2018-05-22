import cv2                              
import numpy as np
import math


# open the camera 
cap = cv2.VideoCapture(0)

# used this variable to save the outputs
count = 0

#while the camera is open perform the following operations 
while( cap.isOpened() ) :
    #read input from camera
    ret,img = cap.read()
    cv2.imwrite('ProjectOutput/image%3d.jpg' % count,img)
    #define a frame for the gesture recognition
    cv2.rectangle(img, (301,301), (99,99), (0,0,255),0)
    #create a new window of the frame 
    mask_window = img[100:300, 100:300]
    #cv2.imwrite('ProjectOutput/''mask_window%3d.jpg' % count,mask_window)
    
    # convert the above window to a gray scale image to perform thresholding
    grey = cv2.cvtColor(mask_window, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    # apply thresholding on the image so that foreground is extracted from the background
    ret, thresh = cv2.threshold(blur,200,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #used to store the contour area of the largest contour
    max_area=0
    max_cnt_number =0
    
    #find all the contours for the thresholded image
    image,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # if contours exist then perform the following operations 
    if contours is not None:
        # check all the contours and find the contour with largest area using the cv2.contourArea() function 
        for cnt in range(len(contours)):
            current_cnt=contours[cnt]
            current_area = cv2.contourArea(current_cnt)
            if(current_area>max_area):
                max_area=current_area
                max_cnt_number = cnt
        max_cnt = contours[max_cnt_number]
        #find the convexity hull of the image
        hull = cv2.convexHull(max_cnt)
        #find the hull's area 
        hull_area = cv2.contourArea(hull)
        #we are using the defects area for recognizing the gesture as we have gestures with similar hull area values
        defects_area = hull_area - max_area
        #display the contours and conexity hull on the blank image contourdisplay 
        contoursdisplay = np.zeros(mask_window.shape,np.uint8)
        cv2.drawContours(contoursdisplay,[max_cnt],0,(0,255,0),2)
        cv2.drawContours(contoursdisplay,[hull],0,(0,0,255),2)
        #find the convexity defects of the image window
        hull = cv2.convexHull(max_cnt, returnPoints=False)
        defects = cv2.convexityDefects(max_cnt, hull)
        count_defects = 0
        cv2.drawContours(contoursdisplay, contours, -1, (255, 0, 0), 3)
        
        #if defects exist
    if defects is not None:
        #find the start,end and farthest points for the defects
        for i in range(len(defects)):
            s,e,f,d = defects[i,0]

            start = tuple(max_cnt[s][0])
            end = tuple(max_cnt[e][0])
            far = tuple(max_cnt[f][0])
            # use the cosine rule for finding the angles between 
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            #check if the angle is less than 90 and then increments the number of triangles in that image
            if angle <= 90:
                count_defects += 1
                #display the triangular points on the contourdisplay image
                cv2.circle(contoursdisplay, far, 1, [255,0,0], -1)
            cv2.line(contoursdisplay,start, end, [0,255,0], 2)
            #cv2.imwrite('ProjectOutput/contour%3d.jpg' % count,contoursdisplay)
            shape_str = ''
        normalized_area=hull_area/max_area
#         categorize the gesture 
        if(normalized_area>1.7 and normalized_area<1.8 and count_defects ==0):
            shape_str = 'Namaste!'
            cv2.putText(img,shape_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            #cv2.imwrite('ProjectOutput/'+str(shape_str)+'%3d.jpg' % count,thresh)
        elif(normalized_area>1.47 and normalized_area<1.55 and count_defects==2):
            shape_str = 'Lapet'
            cv2.putText(img,shape_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            #cv2.imwrite('ProjectOutput/'+str(shape_str)+'%3d.jpg' % count,thresh)
        elif(normalized_area>1.25 and normalized_area<1.35 and count_defects==3):
            shape_str = 'Khoobsurat'
            cv2.putText(img,shape_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            #cv2.imwrite('ProjectOutput/'+str(shape_str)+'%3d.jpg' % count,thresh)
        elif(normalized_area>1.45 and normalized_area<1.55 and count_defects==1):
            shape_str = 'Victory'
            cv2.putText(img,shape_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            #cv2.imwrite('ProjectOutput/'+str(shape_str)+'%3d.jpg' % count,thresh)
#         else:
#         cv2.putText(img,str(normalized_area), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
#         cv2.putText(img,str(count_defects), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    #Display all the windows 
    cv2.imshow('Thresholded',thresh)
    cv2.imshow('input',img)
    cv2.imshow('Contour',contoursdisplay)

    k = cv2.waitKey(10)
    if k == 27:
        break
    count = count + 1
cap.release()
cv2.destroyAllWindows()
    
    
    
