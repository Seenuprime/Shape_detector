import cv2 as cv

cap = cv.VideoCapture("shapes.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, binary_image = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)

    cv.imshow("iamge", binary_image)

    if cv.waitKey(0):
        break

cap.release()
cv.destroyAllWindows()