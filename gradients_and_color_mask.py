import cv2
import numpy as np

def gradients_and_color_mask(calibrated):
    
    # HLS cylinder and Saturation channel
    HLS = cv2.cvtColor(np.copy(calibrated), 
                       cv2.COLOR_RGB2HLS).astype(np.float)
    saturation = HLS[:,:,2]
    
    # sobel_x
    kernel_size = 3
    sobel_x = np.absolute(cv2.Sobel(saturation, cv2.CV_64F, 
                                    1, 0, ksize=kernel_size))
    # scale to range of [0,255] then convert to np.uint8
    sobel_x_uint8 = np.uint8(255 * (sobel_x/np.max(sobel_x)))
    
    # mask_x
    mask_x = np.zeros_like(sobel_x_uint8)
    threshold_low = 20
    threshold_high = 100
    mask_x[(sobel_x_uint8 >= threshold_low) & 
           (sobel_x_uint8 <=threshold_high)] = 1
    
    # sobel_y
    sobel_y = np.absolute(cv2.Sobel(saturation, cv2.CV_64F, 
                                    0, 1, ksize=kernel_size))
    # scale to range of [0,255] then convert to np.uint8
    sobel_y_uint8 = np.uint8(255 * (sobel_y/np.max(sobel_y)))
    
    # mask_y
    mask_y = np.zeros_like(sobel_y_uint8)
    mask_y[(sobel_y_uint8 >= threshold_low) & 
           (sobel_y_uint8 <=threshold_high)] = 1
    
    # sobel_mag
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # scale to range of [0,255] then convert to np.uint8
    sobel_mag_uint8 = np.uint8(255 * (sobel_mag/np.max(sobel_mag)))
    
    # mask_mag
    mask_mag = np.zeros_like(sobel_mag_uint8)
    mask_mag[(sobel_mag_uint8 >= threshold_low) & 
             (sobel_mag_uint8 <=threshold_high)] = 1
    
    # scharr_direction
    scharr_x = np.absolute(cv2.Scharr(saturation, cv2.CV_64F, 1, 0))
    scharr_y = np.absolute(cv2.Scharr(saturation, cv2.CV_64F, 0, 1))
    scharr_direction=np.arctan2(scharr_y,scharr_x)
    
    # mask_direction
    threshold_low_direction = 0.7
    threshold_high_direction = 1.3
    mask_direction = np.zeros_like(scharr_direction)
    mask_direction[(scharr_direction >= threshold_low_direction) & 
                   (scharr_direction <= threshold_high_direction)] = 1
    
    # mask_color
    threshold_low_color = 170
    threshold_high_color = 255
    mask_color = np.zeros_like(saturation)
    mask_color[(saturation >= threshold_low_color) & 
               (saturation <= threshold_high_color)] = 1
    
    # mask_gradient
    mask_gradient = np.zeros_like(saturation)
    mask_gradient[((mask_x==1)&(mask_y==1))| 
                  ((mask_mag==1)&(mask_direction==1))]=1
    
    # mask_total
    mask_total = np.zeros_like(saturation)
    mask_total[(mask_gradient==1)|(mask_color==1)] = 1
    
    return saturation, mask_total
