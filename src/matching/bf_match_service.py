import cv2

def bf_match(des1, des2, cross_check=True):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
