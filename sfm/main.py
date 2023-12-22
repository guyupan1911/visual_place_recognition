import sys

import pypangolin as pango
from OpenGL.GL import *

import cv2
import numpy as np

import visualization

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

    H12inv = np.linalg.inv(H12)
   
    h11inv = H12inv[0][0]
    h12inv = H12inv[0][1]
    h13inv = H12inv[0][2]
    h21inv = H12inv[1][0]
    h22inv = H12inv[1][1]
    h23inv = H12inv[1][2]
    h31inv = H12inv[2][0]
    h32inv = H12inv[2][1]
    h33inv = H12inv[2][2]

    score = 0
    N = len(pts1)
    nInliers = 0
    for i in range(N):
        u1 = pts1[i][0]
        v1 = pts1[i][1]
        u2 = pts2[i][0]
        v2 = pts2[i][1]

        # Reprojection error in first image
        # x2in1 = H12*x2

        w2in1inv = 1.0/(h31*u2+h32*v2+h33)
        u2in1 = (h11*u2+h12*v2+h13)*w2in1inv
        v2in1 = (h21*u2+h22*v2+h23)*w2in1inv

        squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1)
        
        if squareDist1 > 9.0:
            continue
        
        # Reprojection error in second image
        # x1in2 = H21*x1

        w1in2inv = 1.0/(h31inv*u1+h32inv*v1+h33inv)
        u1in2 = (h11inv*u1+h12inv*v1+h13inv)*w1in2inv
        v1in2 = (h21inv*u1+h22inv*v1+h23inv)*w1in2inv

        squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);        

        if squareDist2 > 9.0:
            continue

        print(f'squareDist1: {squareDist1}, squareDist2: {squareDist2}')
        nInliers += 1
        score = score + squareDist1 + squareDist2
    print(f'nInliers: {nInliers}')
    return score / nInliers

def CheckRT(pts3d, pts1, P1, pts2, P2):
    print(f'pts3d: {pts3d.shape} pts1 {pts1.shape} pts2 {pts2.shape}')

    average_error = 0
    N = 0
    for i in range(pts3d.shape[1]):
        if pts3d[3,i] < 0:
            continue
        # reprojection error in im1
        x1 = P1 @ pts3d[:,i]
        x1 = x1 / x1[2]
        x1_gt = pts1[:,i]
        error1 = np.linalg.norm(x1[:2] - x1_gt) 

        # reprojection error in im2
        x2 = P2 @ pts3d[:,i]
        x2 = x2 / x2[2]
        x2_gt = pts2[:,i]
        error2 = np.linalg.norm(x2[:2] - x2_gt) 
        
        average_error += error1 + error2
        N += 1
        # print(f'x1: {x1} x1_gt: {x1_gt} error1: {error1} \nx2: {x2} x2_gt: {x2_gt} error2: {error2}')
    return average_error / N


if __name__=='__main__':
    im1 = cv2.imread('sfm/data/indoor2/0.png')
    im2 = cv2.imread('sfm/data/indoor2/1.png')

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

    # draw matches
    im_match = np.ndarray((1,1))
    im_match = cv2.drawMatches(im1, kpts1, im2, kpts2, good_matches, im_match, flags=2)
    cv2.namedWindow('feature match', cv2.WINDOW_NORMAL)
    cv2.imshow('feature match', im_match)
    cv2.waitKey(0)


    # epipolar geometry
    if 1:
        F, Fmask = cv2.findFundamentalMat(pts2, pts1, cv2.FM_RANSAC, 3, 0.999, 2000)
        print(f'F: {F}, mask: {Fmask.shape}')

        score = CheckFundamental(pts2, pts1, F)
        print(f'score: {score}')

        # recover pose
        E = np.transpose(K) @ F @ K
        _,R,t,mask = cv2.recoverPose(E, pts2, pts1, K)
        print(f'R: {R}, \nt: {t} \nmask: {mask}')
        t = t / np.linalg.norm(t)

        P1 = np.ndarray((3,4))
        P1[:3,:3] = np.identity(3)
        P1[:3,3:] = np.zeros((3,1))
        P1 = K@P1

        P2 = np.ndarray((3,4))
        P2[:3,:3] = np.transpose(R)
        P2[:3,3:] = -np.transpose(R) @ t
        P2 = K@P2
        print(f'P1: {P1} \nP2: {P2}')

        points3d = cv2.triangulatePoints(P1, P2, np.transpose(pts1), np.transpose(pts2))
        for i in range(points3d.shape[1]):
            points3d[:,i] = points3d[:,i] / points3d[3,i]
            if points3d[2,i] < 0:
                points3d[3,i] = -1
        print(f'points3d {points3d}')

        reprojection_error = CheckRT(points3d, np.transpose(pts1), P1, np.transpose(pts2), P2)
        print(f'reprojection_error: {reprojection_error}')

    # homography transform
    else:
        H, mask = cv2.findHomography(pts2, pts1, method=cv2.RANSAC)
        print(f'H: {H}')

        score = CheckHomography(pts1, pts2, H)
        print(f'score: {score}')

        _, Rs, ts, _ = cv2.decomposeHomographyMat(H, K)
        print(f'Rs: {Rs}, \nts: {ts}')

        # for i in range(4):
        R = Rs[1]
        t = ts[1]
        t = t / np.linalg.norm(t)

        P1 = np.ndarray((3,4))
        P1[:3,:3] = np.identity(3)
        P1[:3,3:] = np.zeros((3,1))
        P1 = K@P1

        P2 = np.ndarray((3,4))
        P2[:3,:3] = np.transpose(R)
        P2[:3,3:] = -np.transpose(R) @ t
        P2 = K@P2
        print(f'P1: {P1} \nP2: {P2}')

        points3d = cv2.triangulatePoints(P1, P2, np.transpose(pts1), np.transpose(pts2))
        for i in range(points3d.shape[1]):
            points3d[:,i] = points3d[:,i] / points3d[3,i]
            if points3d[2,i] < 0:
                points3d[3,i] = -1
        print(f'points3d {points3d}')

        reprojection_error = CheckRT(points3d, np.transpose(pts1), P1, np.transpose(pts2), P2)
        print(f'reprojection_error: {reprojection_error}')

    # pangolin
    win = pango.CreateWindowAndBind("main", 640, 480)
    glEnable(GL_DEPTH_TEST)

    pm = pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
    mv = pango.ModelViewLookAt(0,0,5,0,0,0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)

    handler = pango.Handler3D(s_cam)
    d_cam = (
        pango.CreateDisplay()
        .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach.Pix(180),
            pango.Attach(1),
            -640.0 / 480.0,
        )
        .SetHandler(handler)
    )

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        
        # set background color 
        glClearColor(255.0,255.0,255.0,0.0)

        Twc1 = np.identity(4)
        Twc2 = np.identity(4)
        Twc2[:3,:3] = R
        Twc2[:3,3:] = t
        # print(f'Twc1: {Twc1}, Twc2: {Twc2}')
        visualization.DrawCamera(Twc1)
        visualization.DrawCamera(Twc2)
        visualization.DrawCloud(points3d)

        # pango.glDrawColouredCube()
        pango.FinishFrame()