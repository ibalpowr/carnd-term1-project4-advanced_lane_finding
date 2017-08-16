import numpy as np
import cv2

def perspective_transform(image):

    h = image.shape[0]
    w = image.shape[1]

    # provide four point correspondences
    knob = 65   # from trial and error
    y_top = 450   # kind of at where the horizon is
    
    # four points from the source
    # start w/ top left, then clockwise
    src_pt_top_left = (w//2-knob, y_top)
    src_pt_top_right = (w//2+knob, y_top)
    src_pt_bot_right = (w, h)
    src_pt_bot_left = (0,h)
    
    src = np.float32([src_pt_top_left, src_pt_top_right,
                      src_pt_bot_right, src_pt_bot_left])
    
    # four poinst from the destination
    # start w/ top left, then clockwise
    dst_pt_top_left = (knob,0)
    dst_pt_top_right = (w-knob,0)
    dst_pt_bot_right = (w-knob,h)
    dst_pt_bot_left = (knob,h)
    
    dst = np.float32([dst_pt_top_left, dst_pt_top_right,
                      dst_pt_bot_right, dst_pt_bot_left])
    
    # get homography matrix
    H_matrix = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    
    # return warped frontal parallel, H_matrix, H_inv
    return (cv2.warpPerspective(image,H_matrix,(w,h)),H_matrix,H_inv)
