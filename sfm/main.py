import cv2
import numpy as np

# camera intrinstic
K = np.identity(3)
K[0][0] = 2428.58
K[0][2] = 1080
K[1][1] = 2428.58
K[1][2] = 1080

# distortion coefficients
D = np.ndarray(5)
D[0] = -0.32
D[1] = -0.28
D[2] = 0
D[3] = 0
D[4] = 0.6

print(f'K: {K}, \nD: {D}')

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,h,c = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (int(x0),int(y0)), (int(x1),int(y1)), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

if __name__=='__main__':
    im1 = cv2.imread('sfm/data/indoor/query.png')
    im2 = cv2.imread('sfm/data/indoor/true.png')

    im1 = cv2.undistort(im1, K, D)
    im2 = cv2.undistort(im2, K, D)

    # imshow = cv2.hconcat([im0, im1], 1)
    # cv2.namedWindow('undistorted image', cv2.WINDOW_NORMAL)
    # cv2.imshow('undistorted image', imshow)
    # cv2.waitKey(0)
    
    # extract features
    feature_extractor = cv2.SIFT.create(2000)
    kpts1, desc1 = feature_extractor.detectAndCompute(im1, None)
    kpts2, desc2 = feature_extractor.detectAndCompute(im2, None)

    print(f'kpts: {type(kpts1[0])}, desc1: {desc1.shape}')

    # feature match
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf_matcher.knnMatch(desc1, desc2, 2)
    print(f'knn_matches: {type(knn_matches[0][0])}')

    good_matches = []
    pts1 = []
    pts2 = []
    for match in knn_matches:
        if match[0].distance < 0.5 * match[1].distance:
            good_matches.append(match[0])
            pts1.append(kpts1[match[0].queryIdx].pt)
            pts2.append(kpts2[match[0].trainIdx].pt)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    good_matches = tuple(good_matches)
    print(f'good matches: {len(good_matches)}')

    # find ego motion
    F, Fmask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.999, 100)
    print(f'F: {F}, mask: {Fmask.shape}')

    # im_good_matches = cv2.drawMatches(im1, kpts1, im2, kpts2, good_matches, None, flags=2,
    #     matchesMask = mask)
    # cv2.namedWindow('good matches', cv2.WINDOW_NORMAL)
    # cv2.imshow('good matches', im_good_matches)
    # cv2.waitKey(0)

    # draw epipolar line
    lines1 = cv2.computeCorrespondEpilines(pts2, 2,F)
    lines1 = lines1.reshape(-1,3)
    im1,im2 = drawlines(im1,im2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1, 1,F)
    lines2 = lines2.reshape(-1,3)
    im1,im2 = drawlines(im2,im1,lines2,pts2,pts1)

    im_epipolar_lines = cv2.hconcat([im, im2])
    cv2.namedWindow('epipolar lines', cv2.WINDOW_NORMAL)
    cv2.imshow('epipolar lines', im_epipolar_lines)
    cv2.waitKey(0)