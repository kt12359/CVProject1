import cv2
import numpy as np
import sys

def convert_color_space_RGB_to_LMS(img_RGB, lab):
    rgb_to_lms = [[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]]
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            img_LMS[i, j] = np.matmul(rgb_to_lms, img_RGB[i, j])
            # Have to take log of LMS if this is for the Lab conversion
            if lab:
                img_LMS[i,j] = np.log10(img_LMS[i,j])
    return img_LMS

def convert_color_space_LMS_to_RGB(img_LMS):
    lms_to_rgb = [[4.4679, -3.5873, 0.1193],[-1.2186, 2.3809, -0.1624],[0.0497, -0.2439, 1.2045]]
    img_RGB = np.zeros_like(img_LMS, dtype=np.float32)
    height = img_LMS.shape[0]
    width = img_LMS.shape[1]

    for i in range(0, height):
        for j in range(0, width):
            img_RGB[i,j] = np.matmul(lms_to_rgb, img_LMS[i,j])

    return img_RGB

def convert_color_space_BGR_to_RGB(img_BGR):
    #copy BGR in reverse
    img_RGB = img_BGR[..., ::-1].copy()
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    #copy RGB in reverse
    img_BGR = img_RGB[..., ::-1].copy()
    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    lms_to_lab1 = [[1, 1, 1], [1, 1, -2], [1, -1, 0]]
    lms_to_lab2 = [[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(6), 0], [0, 0, 1/np.sqrt(2)]]
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]

    # pass in true so that log10 of result is calculated -- necessary for this conversion
    img_LMS = convert_color_space_RGB_to_LMS(img_RGB, True)

    img_Lab = np.zeros_like(img_RGB,dtype=np.float32)
    for i in range(0, height):
        for j in range(0, width):
            img_Lab[i,j] = np.matmul(lms_to_lab1, img_LMS[i,j])
            img_Lab[i,j] = np.matmul(lms_to_lab2, img_Lab[i,j])

    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    lab_to_lms1 = [[np.sqrt(3)/3, 0, 0], [0, np.sqrt(6)/6, 0], [0, 0, np.sqrt(2)/2]]
    lab_to_lms2 = [[1, 1, 1], [1, 1, -1], [1, -2, 0]]
    height = img_Lab.shape[0]
    width = img_Lab.shape[1]
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)
    for i in range(0, height):
        for j in range(0, width):
            img_LMS[i,j] = np.matmul(lab_to_lms1, img_Lab[i,j])
            img_LMS[i,j] = np.matmul(lab_to_lms2, img_LMS[i,j])
            img_LMS[i,j] = np.power(10, img_LMS[i,j])

    img_RGB = convert_color_space_LMS_to_RGB(img_LMS)
    return img_RGB

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    lms_to_ciecam97s = [[2.00, 1.00, 0.05],[1.00, -1.09, 0.09],[0.11, 0.11, -0.22]]
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)
    # Pass in false so that log10 won't be taken -- unnecessary for this conversion, but important for RGB -> Lab
    img_LMS = convert_color_space_RGB_to_LMS(img_RGB, False)
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            img_CIECAM97s[i,j] = np.matmul(lms_to_ciecam97s,img_LMS[i,j])
    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    cie_to_lms = [[0.32787, 0.32159, 0.20608], [0.32787, -0.63534, -0.18540], [0.32787, -0.15688, -4.53512]]
    img_LMS = np.zeros_like(img_CIECAM97s,dtype=np.float32)
    height = img_CIECAM97s.shape[0]
    width = img_CIECAM97s.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            img_LMS[i, j] = np.matmul(cie_to_lms, img_CIECAM97s[i, j])
    img_RGB = convert_color_space_LMS_to_RGB(img_LMS)
    return img_RGB

def transfer_color(img_src, img_target):

    img_src.astype(float)
    img_target.astype(float)

    # get mean for each color channel of src
    red_mean1 = np.mean(img_src[:,:,0])
    green_mean1 = np.mean(img_src[:,:,1])
    blue_mean1 = np.mean(img_src[:,:,2])

    # get std deviation for each color channel of src
    red_std1 = np.std(img_src[:,:,0])
    green_std1 = np.std(img_src[:,:,1])
    blue_std1 = np.std(img_src[:,:,2])

    # get mean for each color channel of target
    red_mean2 = np.mean(img_target[:,:,0])
    green_mean2 = np.mean(img_target[:,:,1])
    blue_mean2 = np.mean(img_target[:,:,2])

    # get std deviation of each color channel for target
    red_std2 = np.std(img_target[:,:,0])
    green_std2 = np.std(img_target[:,:,1])
    blue_std2 = np.std(img_target[:,:,2])

    # get result of std_dev_tgt/std_dev_src -- this will be used to 'spread out' data points after they've been
    # distributed around 0
    std_red = red_std2/red_std1
    std_green = green_std2/green_std1
    std_blue = blue_std2/blue_std1

    # subtract mean from each color channel of src
    img_result = img_src - np.array([red_mean1, green_mean1, blue_mean1])
    # multiply each color channel by standard deviation
    img_result *= np.array([std_red, std_green, std_blue])
    # add mean of each channel of target to each channel of src
    img_result += np.array([red_mean2, green_mean2, blue_mean2])

    return img_result

def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    img_rgb_source_reversed = convert_color_space_BGR_to_RGB(img_RGB_source)
    img_rgb_target_reversed = convert_color_space_BGR_to_RGB(img_RGB_target)

    # convert to LAB color space
    img_lab_src = convert_color_space_RGB_to_Lab(img_rgb_source_reversed)
    img_lab_target = convert_color_space_RGB_to_Lab(img_rgb_target_reversed)

    # transfer colors from target to source
    img_result = transfer_color(img_lab_src, img_lab_target)

    # convert back to RGB, and then BGR to write out to file
    img_final = convert_color_space_Lab_to_RGB(img_result)
    img_final_displayable = convert_color_space_RGB_to_BGR(img_final)
    return img_final_displayable

def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    # reverse order of color channels
    img_RGB_src_reversed = convert_color_space_BGR_to_RGB(img_RGB_source)
    img_RGB_target_reversed = convert_color_space_BGR_to_RGB(img_RGB_target)

    # transfer colors
    img_result = transfer_color(img_RGB_src_reversed, img_RGB_target_reversed)

    # reverse order of color channels back to bgr to write out to file
    img_final = convert_color_space_RGB_to_BGR(img_result)
    return img_final

def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    # reverse order of color channels
    img_rgb_src_reversed = convert_color_space_BGR_to_RGB(img_RGB_source)
    img_rgb_target_reversed = convert_color_space_BGR_to_RGB(img_RGB_target)

    # convert to CIECAM97s color space
    img_cie_src = convert_color_space_RGB_to_CIECAM97s(img_rgb_src_reversed)
    img_cie_target = convert_color_space_RGB_to_CIECAM97s(img_rgb_target_reversed)

    # transfer colors
    img_result = transfer_color(img_cie_src, img_cie_target)

    # convert back to RGB, then reverse order of color channels to write out to file
    img_result_rgb = convert_color_space_CIECAM97s_to_RGB(img_result)
    img_final = convert_color_space_RGB_to_BGR(img_result_rgb)
    return img_final

def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]

    # ===== read input images
    # img_RGB_source: is the image you want to change the its color
    # img_RGB_target: is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    img_RGB_source = cv2.imread(path_file_image_source, cv2.IMREAD_COLOR)
    img_RGB_target = cv2.imread(path_file_image_target,cv2.IMREAD_COLOR)

    img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    # save image to path_file_image_result_in_Lab
    cv2.imwrite(path_file_image_result_in_Lab, img_RGB_new_Lab)

    img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # save image to path_file_image_result_in_Lab
    cv2.imwrite(path_file_image_result_in_RGB, img_RGB_new_RGB)

    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    # save image to path_file_image_result_in_Lab
    cv2.imwrite(path_file_image_result_in_CIECAM97s, img_RGB_new_CIECAM97s)