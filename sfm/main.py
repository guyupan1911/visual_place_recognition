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

def CheckFundamental(pts1, pts2, F12):
    f11 = F12[0][0]
    f12 = F12[0][1]
    f13 = F12[0][2]
    f21 = F12[1][0]
    f22 = F12[1][1]
    f23 = F12[1][2]
    f31 = F12[2][0]
    f32 = F12[2][1]
    f33 = F12[2][2]

    score = 0
    N = len(pts1)
    nInliers = 0
    for i in range(N):
        u1 = pts1[i][0]
        v1 = pts1[i][1]
        u2 = pts2[i][0]
        v2 = pts2[i][1]

        # Reprojection error in second image
        # l2=F21x1=(a2,b2,c2)

        a2 = f11*u1+f12*v1+f13
        b2 = f21*u1+f22*v1+f23
        c2 = f31*u1+f32*v1+f33

        num2 = a2*u2+b2*v2+c2

        squareDist1 = num2*num2/(a2*a2+b2*b2)

        if squareDist1 > 9.0:
            continue

        # Reprojection error in second image
        # l1 =x2tF21=(a1,b1,c1)

        a1 = f11*u2+f21*v2+f31
        b1 = f12*u2+f22*v2+f32
        c1 = f13*u2+f23*v2+f33

        num1 = a1*u1+b1*v1+c1

        squareDist2 = num1*num1/(a1*a1+b1*b1)
        if squareDist2 > 9.0:
            continue

        # print(f'squareDist1: {squareDist1}, squareDist2: {squareDist2}')
        nInliers += 1
        score = score + squareDist1 + squareDist2
    print(f'nInliers: {nInliers}')
    return score / nInliers

def CheckHomography(pts1, pts2, H12):
    h11 = H12[0][0]
    h12 = H12[0][1]
    h13 = H12[0][2]
    h21 = H12[1][0]
    h22 = H12[1][1]
    h23 = H12[1][2]
    h31 = H12[2][0]
    h32 = H12[2][1]
    h33 = H12[2][2]


if __name__=='__main__':
    im1 = cv2.imread('sfm/data/indoor/query.png')
    im2 = cv2.imread('sfm/data/indoor/true.png')

    im1 = cv2.undistort(im1, K, D)
    im2 = cv2.undistort(im2, K, D)

    imshow = cv2.hconcat([im1, im2], 1)
    cv2.namedWindow('undistorted image', cv2.WINDOW_NORMAL)
    cv2.imshow('undistorted image', imshow)
    cv2.waitKey(0)
    
    # extract features
    feature_extractor = cv2.SIFT.create(2000)
    kpts1, desc1 = feature_extractor.detectAndCompute(im1, None)
    kpts2, desc2 = feature_extractor.detectAndCompute(im2, None)

    print(f'kpts: {type(kpts1[0].pt)}, desc1: {desc1.shape}')

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

    # epipolar geometry
    # F, Fmask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.999, 2000)
    # print(f'F: {F}, mask: {Fmask.shape}')

    # score = CheckFundamental(pts1, pts2, F)
    # print(f'score: {score}')

    # homography transform
    H, mask = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    print(f'H: {H}')