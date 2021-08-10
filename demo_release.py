import argparse
import matplotlib.pyplot as plt
from colorizers import *
import time
import cv2
import sys
from PIL import *

start_time = time.time()
cap = cv2.VideoCapture('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Original Video.mp4')
#cap = cv2.VideoCapture('Original Video.mp4')

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
Frame = []
bboxes2 = []
rgblist = []
count = 0
countnameimg = 0
countnextcouple = 2
objcount = 10
sizeorg = (frameWidth, frameHeight)
hint = np.full((frameHeight,frameWidth,3),(0,0,0),np.uint8)
r = 0
g = 0
b = 0
ret = True
#result = cv2.VideoWriter('D:\My Documents\Grad\Islam\colorization-master\Eccv16\eccv Video.avi',cv2.VideoWriter_fourcc(*'MJPG'), 25, sizeorg)
#result2 = cv2.VideoWriter('D:\My Documents\Grad\Islam\colorization-master/Sigg17/siggraph Video.avi',cv2.VideoWriter_fourcc(*'MJPG'), 25, sizeorg)
#result3 = cv2.VideoWriter('D:\My Documents\Grad\Islam\colorization-master/Sigg17/hints Video.avi',cv2.VideoWriter_fourcc(*'MJPG'), 25, sizeorg)
result = cv2.VideoWriter('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Eccv16/eccv Video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, sizeorg)
result2 = cv2.VideoWriter('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Sigg17/siggraph Video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, sizeorg)
result3 = cv2.VideoWriter('/Users/administrator/BOLLA/Documents/Islam/colorization-master/hints Video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, sizeorg)
while count < frameCount and ret:
    ret, frame = cap.read()
    if count == 0:
        greyscaleframe = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    Frame.append(frame)
    count += 1

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
tracker_type = tracker_types[6]

def createTrackerByName(trackerType):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker


# Exit if video not opened.
if not cap.isOpened():
    print("Could not open video")
    sys.exit()

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()
# Initialize MultiTracker
# Define an initial bounding box
bboxes2 = []
count = 0
currcorners = cv2.goodFeaturesToTrack(greyscaleframe, objcount, 0.01, 10)
currcorners = np.int0(currcorners)
w = 50
h = 50

for i in currcorners:
    x, y = i.ravel()
    bboxes2.append((x, y, w, h))
# Initialize tracker with first frame and bounding box
# ok = tracker.init(Frame[0], bbox)
i = 0
while i < objcount:
    multiTracker.add(createTrackerByName(tracker_type), Frame[count], bboxes2[i])
    i += 1

count = 0
while count < frameCount:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams.jpg')
    parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                        help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    opt = parser.parse_args()
    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if opt.use_gpu:
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    # img = load_img(opt.img_path)
    img = Frame[count]
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if count > 0:
        (hints, hints_rs) = preprocess_img(hint, HW=(256, 256))
        print(hints_rs.shape)
        print(tens_l_rs.shape)
        out_hints_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs, hints_rs).cpu())
        #cv2.imwrite('D:\My Documents\Grad\Islam\colorization-master\hints\hints' + str(countnameimg) + '.png',cv2.cvtColor(out_hints_siggraph17 * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite('/Users/administrator/BOLLA/Documents/Islam/colorization-master/hints/hint' + str(countnameimg) + '.png',cv2.cvtColor((out_img_siggraph17 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        result3.write(cv2.cvtColor((out_hints_siggraph17 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        #temp = Image.open(out_hints_siggraph17)
        #pixels = list(temp.getdata())
        #print(pixels)
    if opt.use_gpu:
        tens_l_rs = tens_l_rs.cuda()
    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs, None).cpu())
    #temp = Image.open(out_img_siggraph17)
    #pixels = list(temp.getdata())
    # plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
    # plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)result.write(cv2.cvtColor((eccv[count]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    cv2.imwrite('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Eccv16/eccv16' + str(countnameimg) + '.png',cv2.cvtColor((out_img_eccv16 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    cv2.imwrite('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Sigg17/siggraph17' + str(countnameimg) + '.png',cv2.cvtColor((out_img_siggraph17 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    #cv2.imwrite('D:\My Documents\Grad\Islam\colorization-master\Eccv16\eccv16' + str(countnameimg) + '.png',cv2.cvtColor(out_img_eccv16 * 255, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('D:\My Documents\Grad\Islam\colorization-master\Siggraph17\siggraph17' + str(countnameimg) + '.png',cv2.cvtColor(out_img_siggraph17 * 255, cv2.COLOR_BGR2RGB))
    '''
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (ECCV 16)')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(out_img_siggraph17)
    plt.title('Output (SIGGRAPH 17)')
    plt.axis('off')
    plt.show()
    '''
    result.write(cv2.cvtColor((out_img_eccv16 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    result2.write(cv2.cvtColor((out_img_siggraph17 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    timer = cv2.getTickCount()

    if count == countnextcouple:
        multiTracker = cv2.MultiTracker.clear(multiTracker)
        bboxes2 = ([tuple(row) for row in bboxes2])
        multiTracker = cv2.MultiTracker_create()

        i = 0
        while i < objcount:
            multiTracker.add(createTrackerByName(tracker_type), Frame[count], bboxes2[i])
            i += 1

        countnextcouple += 2

    ok, bboxes2 = multiTracker.update(Frame[count])
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    hint = np.full((frameHeight, frameWidth, 3), (0, 0, 0), np.uint8)
    for i, newobj in enumerate(bboxes2):
        print(int(newobj[0]), frameWidth, int(newobj[1]), frameHeight)
        # if int(newobj[0]) < frameWidth & int(newobj[1]) < frameHeight:
        print(int(newobj[0]), int(newobj[1]))
        r, g, b = out_img_siggraph17[int(newobj[1]), int(newobj[0])]  # the rgb of the objects
        # rgblist.append(rrr)
        # print('' + str(rgblist))

        p1 = (int(newobj[0]), int(newobj[1]))
        p2 = (int(newobj[0] + newobj[2]), int(newobj[1] + newobj[3]))
        #hint[p1[1], p1[0]] = (r*255, g*255, b*255)
        hint[p1[1], p1[0]] = (255, 0, 0)
        cv2.rectangle(Frame[count], p1, p2, (255, 0, 0), 2, 1)

    #cv2.imwrite('D:\My Documents\Grad\Islam\colorization-master\Prev hints\hints' + str(countnameimg) + '.png',cv2.cvtColor(hint * 255, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Prev hints/hint' + str(countnameimg) + '.png',cv2.cvtColor((hint * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    print(hint)
    cv2.imwrite('/Users/administrator/BOLLA/Documents/Islam/colorization-master/Prev hints/hint' + str(countnameimg) + '.png',cv2.cvtColor(hint, cv2.COLOR_BGR2RGB))
    # Display tracker type on frame
    cv2.putText(Frame[count], tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # Display FPS on frame
    cv2.putText(Frame[count], "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # Display result
    cv2.imshow("Tracking", Frame[count])

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

    count += 1
    countnameimg += 1

cap.release()
print("--- %s seconds ---" % (time.time() - start_time))
