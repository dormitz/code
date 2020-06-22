import numpy as np
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from PIL import Image, ImageFilter
import cv2
from scipy import interpolate
from skimage import io, color, exposure
import pysift
from numpy import savetxt, loadtxt

def getPoints(im1, im2, N):
    p1 = np.zeros((2,N))
    p2 = np.zeros((2, N))
    for i in range(N):
        plt.figure(num=1)
        plt.imshow(im1, cmap='gray')
        plt.axis('off')
        point = plt.ginput(n=2, timeout=0)
        p1[0, i] = point[0][0]
        p1[1, i] = point[0][1]

        plt.figure(num=2)
        plt.imshow(im2,cmap='gray')
        plt.axis('off')
        point = plt.ginput(n=2, timeout=0)
        p2[0, i] = point[0][0]
        p2[1, i] = point[0][1]
    return p1, p2

def computeH(p1, p2):
    m, n = p2.shape
    part1=np.zeros((2,2*n))
    part1[:, ::2] = p2
    tmp=np.ones((1,n))
    part2 = np.zeros((1, 2 * n))
    part2[:, ::2] = tmp

    part3=np.zeros((2,2*n))
    part3[:, 1::2] = p2
    part4 = np.zeros((1, 2 * n))
    part4[:, 1::2] = tmp
    part5=-part1-part3
    part6=-np.reshape(p1.T,(1,-1))
    part5[0,:]=-part5[0,:]*part6
    part5[1, :] = -part5[1, :] * part6

    A=np.concatenate((part1,part2,part3,part4,part5,part6)).T
    #print(A)
    M=np.dot(A.T,A)
    w,v=np.linalg.eig(M)
    w_arg=np.argsort(w)
    h=v[:,w_arg==0]
    H2to1 = np.reshape(h.T, (3, 3))



    return H2to1



def warpH(im1, H, out_size):

    lab1 = color.rgb2lab(im1)

    new_imH = out_size[0]
    new_imW = out_size[1]
    warp_im1 = np.empty([new_imH,new_imW,3])
    H_inv = np.linalg.inv(H)

    grid_x = np.arange(np.shape(lab1)[1])
    grid_y = np.arange(np.shape(lab1)[0])
    kind = 'cubic'
    f1 = interpolate.interp2d(grid_x, grid_y, lab1[:,:,0], kind)
    f2 = interpolate.interp2d(grid_x, grid_y, lab1[:,:,1], kind)
    f3 = interpolate.interp2d(grid_x, grid_y, lab1[:,:,2], kind)

    # Inverse Warping
    for i in range (0,new_imW):
        for j in range (0,new_imH):
            M = H_inv @ np.array([i,j,1])
            x, y = M[0]/M[2], M[1]/M[2]

            if 0 <= x < np.shape(lab1)[1] and 0 < y <= np.shape(lab1)[0]:
                warp_im1[j,i,0] = f1(x,y)
                warp_im1[j,i,1] = f2(x,y)
                warp_im1[j,i,2] = f3(x,y)
    warp_im1 = color.lab2rgb(warp_im1)
    warp_im1 = 255 * warp_im1

    return warp_im1

def imageStitching(img1, wrap_img2):

    panoImg = np.empty([np.shape(wrap_img2)[0], np.shape(wrap_img2)[1], 3])
    panoImg[0:np.shape(img1)[0], 0:np.shape(img1)[1], 0:3] = img1/255

    th = 0.1
    for i in range(0, np.shape(wrap_img2)[0]):
        for j in range(0, np.shape(wrap_img2)[1]):
            if (wrap_img2[i,j,0]>th and wrap_img2[i,j,1]>th and wrap_img2[i,j,2]>th):
                panoImg[i,j,:]=wrap_img2[i,j,:]

    return panoImg

def getPoints_SIFT(im1, im2):

    im1_g = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = pysift.computeKeypointsAndDescriptors(im1_g)
    kp2, des2 = pysift.computeKeypointsAndDescriptors(im2_g)

    MIN_MATCH_COUNT = 10

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.4 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)

    print(good)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, matches, None, **draw_params)
    #
    # plt.imshow(img3, ), plt.show()

    h1, w1 = im1_g.shape
    h2, w2 = im2_g.shape
    hdif = int((h2 - h1) / 2)
    p1 = np.empty((0, 2))
    p2 = np.empty((0, 2))
    for m in good:
        pt1 = np.asarray([[int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif)]])
        pt2 = np.asarray([[int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])]])
        # print(pt1)
        # print(np.shape(pt1))
        p1 = np.append(p1,pt1,axis=0)
        p2 = np.append(p2, pt2,axis=0)

    p1 = np.transpose(p1)
    p2 = np.transpose(p2)
    return p1, p2


def ransacH(p1, p2, nIter, tol):
    rgb1 = io.imread( 'D:\onedrive technion\OneDrive - Technion\semester 6\computer vision\hw\hw4\code\data\incline_L.png')
    rgb2 = io.imread('D:\onedrive technion\OneDrive - Technion\semester 6\computer vision\hw\hw4\code\data\incline_R.png')
    maxinliers=0
    bestH=np.zeros((3,3))
    for k in range (nIter):
        index = np.random.choice(p1.shape[1], 5, replace=False) #determine if we need 4 or 5 points
        p1_maybe = p1[:,index]
        p2_maybe = p2[:,index]
        H = computeH(p1_maybe,p2_maybe)
        p1_rest = np.delete(p1, index, axis=1)
        p2_rest = np.delete(p2, index, axis=1)
        #plotMatches(rgb1, rgb2, p1_rest, p2_rest)
        num_inlier=0
        for i in range(p1_rest.shape[1]):
            M = H @ np.array([p2_rest[0,i], p2_rest[1,i], 1])
            x, y = M[0]/M[2], M[1]/M[2]
            err=np.sqrt((p1_rest[0,i]-x)**2+(p2_rest[1,i]-y)**2)
            if err<=tol:
                p1_maybe = np.append(p1_maybe, p1_rest[:,i,None], axis= 1)
                p2_maybe = np.append(p2_maybe, p2_rest[:,i,None], axis=1)
                num_inlier=num_inlier+1
        if num_inlier>maxinliers:
            maxinliers=num_inlier
            p1_inliars=p1_maybe
            p2_inliars = p2_maybe
            bestH=computeH(p1_maybe,p2_maybe)
            #M = H @ np.array([p2_maybe[0,7], p2_maybe[1,7], 1])
            #x7, y7 = M[0] / M[2], M[1] / M[2]
            #test=test
    print(maxinliers)
    #plotMatches(rgb1, rgb2, p1_inliars, p2_inliars)
    return bestH, p1_inliars,p2_inliars


def plotMatches(im1, im2, p1,p2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1] + im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(100, 50))
    plt.imshow(im, cmap='gray')
    for i in range(p1.shape[1]):
        x = np.asarray([p1[0,i],p1[0,i]+p2[0,i]])
        y = np.asarray([p1[1,i], p2[1,i]])
        plt.plot(x, y, 'r')
        plt.plot(x, y, 'g.')
    plt.show()


def warpPoints(p1,p2,H):
    p2_matches= np.empty((2,0))
    flag=0
    for i in range(p1.shape[1]):
        M = H @ np.array([p2[0, i], p2[1, i], 1])
        x, y = M[0] / M[2], M[1] / M[2]
        p2_matches = np.append(p2_matches, np.array([[x],[y]]), axis=1)
    flag=1
    return p2_matches

def find_borders(w,h,H):
    M = H @ np.array([0, 0, 1])
    x, y = M[0] / M[2], M[1] / M[2]
    min_x=x
    max_x=x
    min_y=y
    max_y=y
    M = H @ np.array([0, 0, 1])
    x, y = M[0] / M[2], M[1] / M[2]
    if x < min_x:
        min_x = x
    if x > min_x:
        max_x = x
    if y < min_y:
        min_y = y
    if y > min_y:
        max_y = y
    return min_x,max_x,min_y,max_y




