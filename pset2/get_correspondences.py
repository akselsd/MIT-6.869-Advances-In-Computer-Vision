import argparse
import colorsys
import numpy as np
import cv2
import sys

WINDOW_NAME = 'Correspondence Matcher. Image1 | Image2'

def parse_args():
    parser = argparse.ArgumentParser(description="Helper to annotate point correspondences. IMAGES MUST BE THE SAME SIZE.")
    parser.add_argument('image1', help='The image to be shown on the left')
    parser.add_argument('image2', help='The image to be shown on the right')
    file_opener = lambda fp: open(fp,'w')
    parser.add_argument("-o","--output", default=sys.stdout, type=file_opener, help="Output to a given file (defaults to standard output)")
    return parser.parse_args()

def get_colors(n):
    all_h = np.linspace(0.0, 0.9, n)
    s = 1.0
    v = 0.9
    rgb = np.array([colorsys.hsv_to_rgb(h, s, v) for h in all_h])
    rgb = (rgb * 255).astype(np.uint8)
    rgb = [tuple(map(int, rgb[i])) for i in range(len(rgb))]
    return rgb

def render_image(imgA, imgB, kpA, kpB):
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    assert hA == hB, wA == wB
    # Render them both to BGR if gray
    colorA = imgA
    colorB = imgB
    if len(imgA.shape) == 2:
        colorA = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)

    if len(imgB.shape) == 2:
        colorB = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)

    rendered = np.concatenate([colorA, colorB], axis=1)

    n_colors = max(len(kpA), len(kpB)) + 1
    cmap = get_colors(n_colors)

    radius = 3
    thickness = 3

    for i in range(len(kpA)):
        (x,y) = kpA[i]
        cv2.circle(rendered, (x,y), radius, color=cmap[i], thickness=3)

    for i in range(len(kpB)):
        (x,y) = kpB[i]
        cv2.circle(rendered, (x + wA,y), radius, color=cmap[i], thickness=3)
           
    # Now, draw lines
    for i in range(min(len(kpA), len(kpB))):
        (x1, y1) = kpA[i]
        (x2, y2) = kpB[i]
        # When rendered side by side, offset the 2nd x by the width of the left image
        x2 += wA
        cv2.line(rendered, (x1, y1), (x2, y2), color=cmap[i], thickness = 1)


    cv2.imshow(WINDOW_NAME, rendered)


class BooleanHolder():
    def __init__(self, initVal):
        self.val = initVal
    def toggle(self):
        self.val = not self.val
    def get(self):
        return self.val

def main():
    keypoints_a = [(0,0)]
    keypoints_b = []
    
    curr_is_a = BooleanHolder(True)

    print("Left click to mark a point. Image A first, then image B. Z to undo")
    
    args = parse_args()
    imgA = cv2.imread(args.image1)
    imgB = cv2.imread(args.image2)
    W = 600

    height, width, depth = imgA.shape
    imgScale = W/width
    newX,newY = imgA.shape[1]*imgScale, imgA.shape[0]*imgScale

    imgA = cv2.resize(imgA,(int(newX),int(newY)))
    imgB = cv2.resize(imgB,(int(newX),int(newY)))

    print(imgA.shape)
    print(imgB.shape)

    h,w = imgA.shape[:2]
    assert imgA.shape[:2] == imgB.shape[:2], "images must be the same size"

    render_image(imgA, imgB, keypoints_a, keypoints_b)

    def mouse_callback(*args):
        event, mouse_x, mouse_y, _, _ = args
        curr_list = keypoints_a if curr_is_a.get() else keypoints_b
        target_x = mouse_x
        target_y = mouse_y
        if not curr_is_a.get():
            # offset mouse by image width
            target_x -= w
        if target_x >= 0 and target_x < w and target_y >= 0 and target_y < h:
            # valid position
            curr_list[-1] = (target_x, target_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            curr_is_a.toggle()
            curr_list = keypoints_a if curr_is_a.get() else keypoints_b
            curr_list.append((0,0))

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    while True:
        render_image(imgA, imgB, keypoints_a, keypoints_b)
        keypresses = cv2.waitKey(1) & 0xFF
        if keypresses == ord('q'):
            break
        if keypresses == ord('z'):
            # Undo
            print("Undo")
            pass

    curr_list = keypoints_a if curr_is_a.get() else keypoints_b
    curr_list.pop()

    print("Output format: KeypointsA, then KeypointsB")
    print(keypoints_a, file=args.output)
    print(keypoints_b, file=args.output)

    cv2.imwrite(args.image1[:-4] + "res.jpg", imgA)
    

if __name__ == "__main__":
    main()