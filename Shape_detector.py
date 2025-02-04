import cv2 as cv
import numpy as np

cap = cv.VideoCapture("shapes.mp4")
# cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = int(600 / aspect_ratio)
    new_width = int(new_height * aspect_ratio) 

    frame = cv.resize(frame, (new_width, new_height))
    # frame = cv.flip(frame, -1)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # blurred_frame = cv.GaussianBlur(gray_frame, (3,3), 0)
    blurred_frame = cv.medianBlur(gray_frame, 3, 0)

    edges = cv.Canny(blurred_frame, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            ## Approximate the Contour
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)

            # print(approx)

            # Getting the bounding boxs
            x, y, w, h = cv.boundingRect(approx)

            # Classify the shapes based on number of corners
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                square_aspect = w / h 
                shape = "Square" if 0.95 < square_aspect < 1.05 else "Rectangle"
            elif len(approx) > 6:
                shape = "Circle"
            else:
                shape = "Polygon"

            cv.putText(frame, shape, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            

    cv.imshow("Original", frame)
    cv.imshow("Frames", edges)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
