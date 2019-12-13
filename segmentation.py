import os
import copy
import argparse
import re
import math
import numpy as np
import cv2
import pytesseract
import maskrcnn.mrcnn.model as modellib
from maskrcnn.samples.coco import coco
from maskrcnn.mrcnn import utils
from fuzzywuzzy import process

MODEL_DIR = os.path.join('./', "logs")
COCO_MODEL_PATH = os.path.join('./', "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join('./', 'images')
TESSERACT_CONFIG_1 = "-l eng --tessdata-dir ./tesseract --oem 1 --psm 6"
TESSERACT_CONFIG_2 = "-l eng --tessdata-dir ./tesseract --oem 1 --psm 11"


# PARTS OF THIS CODE RELATED TO MASK R-CNN ARE ADAPTED FROM https://github.com/matterport/Mask_RCNN

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    PRE_NMS_LIMIT = 6000


# Checks if coco model exists and downloads it if it doesn't exist.
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = InferenceConfig()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# This list will contain all the airlines in the world. It will be populated by the main function.
AIRLINES = []


def main():
    parser = argparse.ArgumentParser(description='EECS 4422 Project - Extracting Information from Commercial Aircrafts')
    parser.add_argument('--single_image', action='store_true',
                        help='Choose this option with --image_path to work on a single image file.')
    parser.add_argument('--multiple_images', action='store_true',
                        help='Choose this option with --images_path to work on multiple images.')
    parser.add_argument('--image_path', default='./images/tc-jjz.jpg', help='Path for the image file.')
    parser.add_argument('--images_path', default='./images', help='Path for the folder containing image folders.')
    parser.add_argument('--verbose', action='store_true', help='Choose this option to have intermediary images saved.')

    args = parser.parse_args()
    single_image = args.single_image
    multiple_images = args.multiple_images
    image_path = args.image_path
    images_path = args.images_path
    verbose = args.verbose

    global AIRLINES
    AIRLINES = load_airlines()

    if single_image and multiple_images:
        print(
            "You may not choose the single image and multiple images options together. Please choose either one of them.")
        exit(-1)

    # Creates the tmp folder if it does not exist.
    if not os.path.isdir('tmp'):
        os.makedirs('tmp')

    if single_image:
        airline_results, reg_results = process_single_image(image_path, verbose)
        print(airline_results)
        print(reg_results)
        exit(0)

    if multiple_images:
        process_multiple_images(images_path, verbose)
        exit(0)


# Processes a given image and retuns the results.
def process_single_image(image_name, save_files=False):
    results = segment_image(image_name)
    extracted_airplane, boxes, N = extract_airplane(image_name, results['masks'], results['rois'])
    original_extracted_airplane = copy.deepcopy(extracted_airplane)
    original_extracted_airplane = cv2.cvtColor(original_extracted_airplane, cv2.COLOR_BGR2RGB)
    if save_files:
        save_file(extracted_airplane, image_name, '_segmented')

    # Gets the coordinates for the bounding box that has the masked airplane.
    y1, x1, y2, x2 = boxes[0]
    bbox_height = y2 - y1
    bbox_width = x2 - x1

    extracted_airplane = cv2.cvtColor(extracted_airplane, cv2.COLOR_BGR2RGB)
    reg, reg_boxes = extract_reg(copy.deepcopy(extracted_airplane), bbox_height, bbox_width, image_name, save_files)
    reg_results = run_ocr(original_extracted_airplane, reg_boxes)
    airline_results = extract_airline(original_extracted_airplane, boxes[0], results['masks'])
    return str(airline_results[0]), reg_results


# Processes all the images in a given image folder path and saves the results to a text file.
def process_multiple_images(images_path, save_files=False):
    for root, dirs, files in os.walk(images_path, topdown=False):
        for file in files:
            print('Processing ' + file)
            airline_results, reg_results = process_single_image(root + '/' + file, save_files)
            with open('results.txt', 'a') as myfile:
                myfile.write(file + '\n')
                myfile.write('Possible Airline:')
                myfile.write(airline_results + '\n')
                myfile.write('Possible registration codes: (Sorted from highest probability to lowest)')
                for each in reg_results:
                    myfile.write(each + '\n')
                myfile.write('-----------------------------\n')


# Applies thresholding and morphology on a given image and finds the contours on it.
def pre_extract_info(extracted_airplane, iterations, threshold, low, high):
    ret, mask = cv2.threshold(extracted_airplane, low, high, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(extracted_airplane, extracted_airplane, mask=mask)
    ret, new_img = cv2.threshold(image_final, low, high, threshold)  # for black text , cv.THRESH_BINARY_INV
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(new_img, kernel, iterations=iterations)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, dilated


# Filters agiven set of contours using heuristics sushc as aspect ratio.
def find_candidate_bboxes(extracted_airplane, contours, plane_bbox_height, plane_bbox_width, extracted_airplane_bbox):
    results = []
    for contour in contours:
        # Gets rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        # Aspect ratio of the candidate bbox that might include the registration number.
        aspect_ratio = float(w / h)
        if aspect_ratio < 2.50 or aspect_ratio > 5.50:
            continue
        # The ratio of the candidate registration bbox to the bbox around the airplane.
        reg_plane_height_ratio = float(h / plane_bbox_height) * 100
        # The ratio of the candidate registration bbox to the bbox around the airplane.
        reg_plane_width_ratio = float(w / plane_bbox_width) * 100
        if reg_plane_height_ratio > 6.0 or reg_plane_height_ratio < 3.0:
            continue
        if reg_plane_width_ratio > 6.0 or reg_plane_width_ratio < 2.0:
            continue

        # Draws a rectangle around the candidate bbox on the original image.
        cv2.rectangle(extracted_airplane_bbox, (x, y), (x + w, y + h), (255, 0, 255), 2)
        y1, x1, y2, x2 = y, x, y + h, x + w
        results.append([y1, x1, y2, x2])

    return results, extracted_airplane_bbox


# Extracts the bounding boxes that possibly include the registration boxes.
def extract_reg(extracted_airplane, height, width, original_name, save_files=False):
    # Tries increasing dilation if no registration number detected.
    extracted_airplane_bbox = copy.deepcopy(extracted_airplane)
    results_bbox = []
    thresh_lows = [180]
    thresh_highs = [255]
    # Splits the image into its channels.
    b, g, r = cv2.split(extracted_airplane)
    channels = [b, g, r]
    for index, channel in enumerate(channels):
        for i in range(0, 4):
            for low, high in zip(thresh_lows, thresh_highs):
                contours, dilated = pre_extract_info(channel, i, cv2.THRESH_BINARY, low, high)
                contours_inv, dilated_inv = pre_extract_info(channel, i, cv2.THRESH_BINARY_INV, low, high)
                results, extracted_airplane_bbox = find_candidate_bboxes(channel, contours, height, width,
                                                                         extracted_airplane_bbox)
                results_inv, extracted_airplane_bbox = find_candidate_bboxes(channel, contours_inv, height, width,
                                                                             extracted_airplane_bbox)
                if len(results) > 0:
                    results_bbox.append(results)
                if len(results_inv) > 0:
                    results_bbox.append(results_inv)
                if save_files:
                    save_file(dilated, original_name,
                              '_dilated_' + str(i) + '_' + str(low) + '_' + str(high) + '_' + str(index))
                    save_file(dilated_inv, original_name,
                              '_dilated_inv_' + str(i) + '_' + str(low) + '_' + str(high) + '_' + str(index))
    return extracted_airplane_bbox, results_bbox


# Extracts the airline name from a masked image.
def extract_airline(extracted_airplane, bbox, mask):
    # Crop the image.
    y1, x1, y2, x2 = bbox
    cropped_mask = mask[y1:y2, x1:x2, 0]
    width = x2 - x1

    # Using the bounding boxes, discard some unrelated picture of the image i.e black background.

    # This loop squuezes the image from the top.
    for index, each in enumerate(cropped_mask):
        c = np.count_nonzero(each == True)
        if (width / 100 * 50) <= c:
            upper = index
            break
    # This loop squuezes the image from the bottom.
    for index, each in enumerate(np.flip(cropped_mask)):
        c = np.count_nonzero(each == True)
        if (width / 100 * 50) <= c:
            lower = index
            break

    upper = int(upper * 0.75)
    cropped_airplane = extracted_airplane[y1 + upper:y2 - lower, x1:x2]
    cropped_airplane = cv2.cvtColor(cropped_airplane, cv2.COLOR_RGB2GRAY)

    cv2.imwrite('croppedairplane.png', cropped_airplane)
    output = pytesseract.image_to_string(cropped_airplane, config=TESSERACT_CONFIG_2)
    # Cleans the output and splits it into the words detected. Then creates a string and appends each of the words
    # iteratively and also at each iteration compares the levenshtein distance of the string with all the airline names.
    output_stripped = output.strip()
    output_stripped_without_newlines = output_stripped.replace('\n', ' ')
    output_splitted = output_stripped_without_newlines.split(" ")
    output_splitted_filtered = [each for each in output_splitted if each.isalnum()]
    str = ""
    results = []
    for each in output_splitted_filtered:
        str += each
        results.append(process.extractOne(str, AIRLINES))
    # Returns the airline name that occurs most frequently.
    if len(results):
        return max(set(results), key=results.count)
    else:
        return ['No airline results was found']


# Adapted from MaskRCNN github page guide.
def segment_image(image_name):
    image = cv2.imread(image_name, 1)
    results = model.detect([image], verbose=1)
    return results[0]


# Extracts the airplane from given masks and returns the new masked image.
def extract_airplane(image_name, masks, boxes, draw_bbox=False):
    N = boxes.shape[0]
    image = cv2.imread(image_name, 1)
    alpha = 0.5
    for i in range(N):
        if masks is not None:
            mask = masks[:, :, 0]
            for c in range(3):
                image[:, :, c] = np.where(mask == 0, 0, image[:, :, c])
        if draw_bbox and boxes is not None:
            y1, x1, y2, x2 = boxes[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return image, boxes, N


# Runs OCR on the bounding boxes found. Mitigates viewing angle issue and rotates the cropped image.
def run_ocr(image, bboxes):
    # Convert the image to grayscale for better OCR performance.
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    results = []
    for bbox_list in bboxes:
        for bbox in bbox_list:
            y1 = bbox[0] - 20
            x1 = bbox[1] - 20
            y2 = bbox[2] + 20
            x2 = bbox[3] + 20
            # Crop to the part of the image that possible has the registration number.
            cropped = grayscale[y1:y2, x1:x2]
            # We are enlarging the cropped image because it is easier to work on the image this way.
            cropped = cv2.resize(cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

            # We need to do some thresholding for our preprocessing operations. For that, we need to know about the
            # background color. Since we already converted the image to grayscale and cropped the image to a very
            # small size, we can assume that the background will nearly have single color and it will be either black
            # or white. Below, we do a color analysis and guess the color of the background.

            # Normalizing the contrast of the cropped area for better OCR performance.
            cropped = normalize_contrast(cropped)

            # Converting the cropped image to a 1-D array so that we can sum all the pixels in it easily.
            cropped_flattened = cropped.flatten()
            # Summing all pixels in the cropped area.
            sum = np.sum(cropped_flattened)
            # Getting the dimensions of the cropped area and calculating the number of pixels for the analysis.
            height, width = cropped.shape  # CHANGED THIS
            pixel_count = width * height
            # We find the average color value per pixel. If this value is closer to 255, we can assume the background is
            # white, if it is closer to 0 then we can assume it is black.
            sum_avg = sum / pixel_count
            white_diff = abs(255 - sum_avg)
            black_diff = abs(0 - sum_avg)
            background = 'white' if black_diff > white_diff else 'black'

            # We now know the background color and can choose the appropriate thresholding type.
            thresholding_type = cv2.THRESH_BINARY_INV if background == 'white' else cv2.THRESH_BINARY
            thresh = cv2.threshold(cropped, 0, 255, thresholding_type | cv2.THRESH_OTSU)[1]

            # Unfortunately, Tesseract performs poorly when the text is skewed or the viewing angle is not parallel
            # to the text. Therefore, we need to rotate the text so that it aligns horizontally and flatten the text
            # as if the image is taken from a viewing point that is facing the text straight.
            # Aligning the text horizontally is easy. But flattening it is not. To flatten it we will be doing a
            # four point transformation (i.e. warping).
            # But we need reference points to do warping and unfortunately the bounding box coordinates we have
            # does not account for the perspective of the text.

            # Before doing warping we need to rotate the text. Below we first find the angle of the image calculate a
            # rotation matrix and rotate the image by doing an affine transformation.
            # Adapted from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = cropped.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(cropped, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            height, width = rotated.shape
            hough_thresh = int(height / 8)
            edges = cv2.Canny(rotated, 80, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)

            degrees = []
            for i in range(len(lines)):
                for rho, theta in lines[i]:
                    # The lines we are looking for generally have an angle greater than 140. That is why we are skipping
                    # those below this threshold.
                    if math.degrees(theta) < 140:
                        continue
                    degrees.append(math.degrees(theta))

            # Runs OCR on the rotated image before applying four point transformation incase something goes wrong.
            text = pytesseract.image_to_string(rotated, config=TESSERACT_CONFIG_1)
            results.append(text)

            # Parts of code is adapted from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/.
            if len(degrees):
                # We now know the the tilt of the letters. Using this we can estimate the appropiate points for warping.
                degree = np.sum(degrees) / len(degrees)
                degree -= 90  # AYAR CEK

                # Below 4 points are the corners of our bounding box.
                top_left = [x1, y1]
                top_right = [x2, y1]
                bottom_left = [x1, y2]
                bottom_right = [x2, y2]

                # Since the top points and the bottom points are aligned vertically, we can find the distance between
                # the top points and bottom points by finding the difference in y-axis.
                dist_between_tl_bl = (bottom_left[1] - top_left[1])
                shift_distance = dist_between_tl_bl / (math.tan(math.radians(degree)))

                # Applying some trigonometry, we find the shift distance here and shift the points on the x-axis
                # accordingly.
                shifted_bottom_left = [int(x1 + shift_distance), y2]
                shifted_bottom_right = [int(x2 + shift_distance), y2]

                width_a = np.sqrt(((shifted_bottom_right[0] - shifted_bottom_left[0]) ** 2) + (
                        (shifted_bottom_right[1] - bottom_left[1]) ** 2))
                width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
                max_width = max(int(width_a), int(width_b))
                height_a = np.sqrt(
                    ((top_right[0] - shifted_bottom_right[0]) ** 2) + ((top_right[1] - shifted_bottom_right[1]) ** 2))
                height_b = np.sqrt(
                    ((top_left[0] - shifted_bottom_left[0]) ** 2) + ((top_left[1] - shifted_bottom_left[1]) ** 2))
                max_height = max(int(height_a), int(height_b))

                dst_pts = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]], dtype=np.float32)

                # Below 4 lines, we do the warping to flatten the text.
                src_pts = np.array([top_left, top_right, shifted_bottom_right, shifted_bottom_left], dtype=np.float32)
                # dst_pts = np.array([[0, 0], [200, 0], [200, 50], [0, 50]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warp = cv2.warpPerspective(grayscale, M, (max_width, max_height))

                # We had to use the unedited image for the warping. Therefore, the changes we have made before the warping
                # operation is no longer present. But, since we already know the angle from the previous time
                # we rotated the image, we can simply rotate the warped image, and also do the contrast normalization.
                (h, w) = warp.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Tesseract works better with bigger images, that is why we are enlarging our warped image here.
                # We use cubic interpolation here to minimize the quality loss from enlarging the warped image.
                warped_rotated = cv2.warpAffine(warp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                warped_enlarged = cv2.resize(warped_rotated, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

                # We normalize the contrast of the warped image for better OCR performance here.
                warped_improved = normalize_contrast(warped_enlarged)

                # Finally, we send the preprocessed image to OCR and get the results.
                text = pytesseract.image_to_string(warped_improved, config=TESSERACT_CONFIG_1)
                results.append(text)
    return filter_reg_results(results)


# Filters the candidate results for registration codes received from OCR.
# Output is sorted from highest probability to lowest.
def filter_reg_results(results):
    # Below loop goes through each candidate result and gets the ones that are purely alphanumeric.
    # But there is a catch here. Alphanumeric sequences separated with a '-' will not count as alphanumeric.
    # N12345 -> Alphanumeric
    # C-FKGH -> Not alphanumeric
    # This is why we use regex here and get all results that look like the above two.
    filtered = []
    for result in results:
        regex = "([a-zA-Z1-9]+([—]|[-]|[_])+[a-zA-Z1-9]+|[a-zA-Z1-9]+)"
        matches = re.findall(regex, result)
        for match in matches:
            filtered.append(match[0])

    # We discard results with length less than than 3.
    filtered = list(filter(lambda x: len(x) >= 3, filtered))

    # It is likely we will have same results detected repeatedly. Therefore we put all of these in buckets.
    # Also OCR usually detects the dashes incorrectly. Therefore we get the alphanumeric parts of candidate registration
    # codes and build it again to make sure the dash added properly.
    # After this look both data structures are going to look like the following.
    # buckets_with_dash = {'TC-JJZ': 3,}
    # buckets_without_dash = {'Roa' : 2, 'BEE':1}
    buckets_with_dash = {}
    buckets_without_dash = {}
    for each in filtered:
        # If each does not have a dash in it go into here. (N12345) Else go to the else block. (C-FKGH)
        if each.isalnum():
            if each in buckets_without_dash:
                buckets_without_dash[each] += 1
            else:
                buckets_without_dash[each] = 1
        else:
            regex = r"[a-zA-Z1-9]+"
            matches = re.findall(regex, each)
            cleaned_result = matches[0] + '-' + matches[1]
            if cleaned_result in buckets_with_dash:
                buckets_with_dash[cleaned_result] += 1
            else:
                buckets_with_dash[cleaned_result] = 1
    # It is likely candidate result that appears the most frequent is the one actual result. Therefore we list the
    # bucket based on the number of times each result shows up. But if we have any results that contain any dashes
    # then it is VERY likely that is in fact the one we are looking for. That's why we are going to sort both buckets
    # first and prioritize the results with dashes.
    candidates_with_dash = list(buckets_with_dash.keys())
    candidates_without_dash = list(buckets_without_dash.keys())
    candidates_with_dash.sort(reverse=True, key=lambda x: buckets_with_dash[x])
    candidates_without_dash.sort(reverse=True, key=lambda x: buckets_without_dash[x])
    return candidates_with_dash + candidates_without_dash


# Normalizes the contrast of a given image.
def normalize_contrast(img):
    pmax = np.amax(img)
    pmin = np.amin(img)
    lmax = 255
    lmin = 0
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            p = img[y, x]
            p_prime = ((p - pmin) * ((lmax - lmin) / (pmax - pmin))) + lmin
            img[y, x] = p_prime
    return img


# Loads the file that includes all the airlines' names into memory.
def load_airlines(path='airline_list.txt'):
    airlines = []
    try:
        with open(path) as airline_list:
            for line in airline_list:
                airlines.append(line.strip())
        return airlines
    except FileNotFoundError:
        print(
            'airline_list.txt does not exist. Please make sure it exists or generate a new one using scrape.py and try again.')
        exit(-2)


# Saves a given image.
def save_file(image, original_name, suffix):
    file_name_without_ext = os.path.splitext(original_name)[0] + suffix
    file_extension = os.path.splitext(original_name)[1]
    final_file_name = file_name_without_ext + file_extension
    cv2.imwrite('./tmp/' + os.path.basename(final_file_name), image)


# Entrypoint of the script.
if __name__ == '__main__':
    main()
