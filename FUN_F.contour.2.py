import cv2
from imutils.perspective import four_point_transform

def contour_image(image_name, backg=10):
    approxa = []
    border_thickness = 1
    image0 = cv2.imread(image_name)  # Read the image

    height, width, channels = image0.shape
    # Calculate the number of pixels
    num_pixels = height * width

    if num_pixels < 1000000:
        factor = 3
        new_size = (image0.shape[1] * factor, image0.shape[0] * factor)
        image = cv2.resize(image0, new_size, interpolation=cv2.INTER_CUBIC)
    else:
        image = image0
    image = cv2.copyMakeBorder(image, border_thickness, border_thickness, border_thickness, border_thickness,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width, channels = image.shape
    gnum_pixels = height * width

    adaptive_thresholded_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 21, 0)
    adaptive_thresholded_image0 = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, 21, backg)

    contours, _ = cv2.findContours(adaptive_thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    image_with_contours = cv2.cvtColor(adaptive_thresholded_image, cv2.COLOR_GRAY2BGR)
    perimeter = cv2.arcLength(cnt1, True)
    epsilon = 0.02 * perimeter
    approxa = cv2.approxPolyDP(cnt1, epsilon, True)

    if cv2.contourArea(cnt1) >= gnum_pixels * 0.96:
        adaptive_thresholded_image = adaptive_thresholded_image[:adaptive_thresholded_image.shape[0]-5, :]
        contours, _ = cv2.findContours(adaptive_thresholded_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cnt1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        image_with_contours = cv2.cvtColor(adaptive_thresholded_image, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(image_with_contours, [cnt1], -1, (0, 0, 255), 2)

    if cv2.contourArea(cnt1) >= gnum_pixels / 5:
        perimeter = cv2.arcLength(cnt1, True)
        epsilon = 0.02 * perimeter
        approxx = cv2.approxPolyDP(cnt1, epsilon, True)
        contoured = four_point_transform(adaptive_thresholded_image0, approxx.reshape(4, 2) * 1)
    else:
        contoured = adaptive_thresholded_image0
        approx = approxa

    f_contourred = "f_contourred.jpg"
    cv2.imwrite(f_contourred, contoured)  # Save the new image

    return contoured, f_contourred, approxx
