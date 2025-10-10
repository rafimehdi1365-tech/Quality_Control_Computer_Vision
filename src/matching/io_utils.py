import cv2

def draw_matches(img1, kp1, img2, kp2, matches, max_draw=20):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_draw], None, flags=2)
