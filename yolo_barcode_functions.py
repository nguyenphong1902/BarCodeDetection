import cv2 as cv
import math
from pyzbar import pyzbar


def RotateImg(imgx):
    # Convert to greyscale image
    img_gray = cv.cvtColor(imgx, cv.COLOR_BGR2GRAY)

    # Use fast line detector to detect lines from barcode
    fld = cv.ximgproc.createFastLineDetector()
    lines = fld.detect(img_gray)

    # Predict rotation angle of the barcode
    lenSum = [0] * 180
    maxLen = 0
    maxAngle = 0
    for line in lines:
        if line[0, 2] - line[0, 0] == 0:
            continue
        angle = math.floor(
            math.atan((line[0, 3] - line[0, 1]) / (line[0, 2] - line[0, 0])) * 180 / math.pi)
        length = math.sqrt(math.pow(line[0, 3] - line[0, 1], 2) + math.pow(line[0, 2] - line[0, 0], 2))
        lenSum[angle + 89] += length
        if lenSum[angle + 89] > maxLen:
            maxLen = lenSum[angle + 89]
            maxAngle = angle

    return rotate_im(imgx, 90 + maxAngle)

# rotate image an angle without cropping
def rotate_im(image, angle):
    image_height = image.shape[0]
    image_width = image.shape[1]
    diagonal_square = (image_width * image_width) + (
            image_height * image_height
    )
    #
    diagonal = round(math.sqrt(diagonal_square))
    padding_top = round((diagonal - image_height) / 2)
    padding_bottom = round((diagonal - image_height) / 2)
    padding_right = round((diagonal - image_width) / 2)
    padding_left = round((diagonal - image_width) / 2)
    padded_image = cv.copyMakeBorder(image,
                                     top=padding_top,
                                     bottom=padding_bottom,
                                     left=padding_left,
                                     right=padding_right,
                                     borderType=cv.BORDER_CONSTANT,
                                     value=(255, 255, 255)
                                     )
    padded_height = padded_image.shape[0]
    padded_width = padded_image.shape[1]
    transform_matrix = cv.getRotationMatrix2D(
        (padded_height / 2,
         padded_width / 2),  # center
        angle,  # angle
        1.0)  # scale
    rotated_image = cv.warpAffine(padded_image,
                                  transform_matrix,
                                  (diagonal, diagonal),
                                  flags=cv.INTER_LANCZOS4,
                                  borderValue=(255, 255, 255))
    return rotated_image

# Decode image with ZBar
def ScanImg(imgx):
    scanedImg = imgx.copy()
    barcodes = pyzbar.decode(scanedImg)

    if len(barcodes) == 0:
        return -1
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv.rectangle(scanedImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # draw the barcode data and barcode type on the image
        text = "{} ({})".format(barcodeData, barcodeType)
        cv.putText(scanedImg, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 255), 2)
        # print the barcode type and data to the terminal
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    # show the output image
    cv.imshow(text, scanedImg)
    cv.waitKey(1)
