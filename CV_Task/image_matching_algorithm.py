# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch import nn
import os
import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

def compute_keypoints(image_pair):
    """
    Compute keypoints and descriptors for a pair of images using SIFT.

    Args:
    image_pair (tuple): A tuple containing two images (left and right).

    Returns:
    keypoints1, descriptors1, keypoints2, descriptors2: Keypoints and descriptors for the two images.
    """
    sift = cv2.SIFT_create()  # Create the SIFT detector
    left = image_pair[0]  # Left image
    right = image_pair[1]  # Right image

    # Convert images to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray_left, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_right, None)

    return keypoints1, descriptors1, keypoints2, descriptors2


def match_images(descriptors1, descriptors2, trsh=0.8):
    """
    Match descriptors between two sets using the BFMatcher with knnMatch and ratio test.

    Args:
    descriptors1 (ndarray): Descriptors from the first image.
    descriptors2 (ndarray): Descriptors from the second image.
    trsh (float): The ratio threshold for good matches (default 0.8).

    Returns:
    good_matches (list): A list of good matches based on the ratio test.
    """
    descriptors1 = np.float32(descriptors1)  # Convert descriptors to float32
    descriptors2 = np.float32(descriptors2)  # Convert descriptors to float32

    bf = cv2.BFMatcher()  # Create the brute force matcher

    # Perform knnMatch to get the best 2 matches for each descriptor
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []  # List to store good matches
    for m, n in matches:
        if m.distance < trsh * n.distance:  # Apply ratio test
            good_matches.append(m)

    return good_matches


def visualize_keypoints_matches_without_ransac(image1, image2, keypoints1, keypoints2, matches):
    """
    Visualize keypoint matches between two images.

    Args:
    image1 (ndarray): First image.
    image2 (ndarray): Second image.
    keypoints1 (list): List of keypoints for the first image.
    keypoints2 (list): List of keypoints for the second image.
    matches (list): List of good matches.

    Displays the images with matches drawn.
    """
    # Draw the matches on the images
    output_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                   matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.axis('off')
    plt.show()

def ransac_filter(matches, keypoints1, keypoints2, threshold=5.0):
    """
    Filter matches using RANSAC to remove outliers by computing homography.

    Args:
    matches (list): List of good matches.
    keypoints1 (list): List of keypoints for the first image.
    keypoints2 (list): List of keypoints for the second image.
    threshold (float): Threshold for RANSAC algorithm (default 5.0).

    Returns:
    inliers (list): List of inlier matches.
    outliers (list): List of outlier matches.
    """
    if len(matches) < 4:  # Check if there are enough matches for RANSAC
        print("Not enough matches to compute homography.")
        return [], []

    # Extract points from keypoints based on matches
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

    # Find homography using RANSAC
    _, mask = cv2.findHomography(points1, points2, cv2.RANSAC, threshold)

    # Separate inliers and outliers based on the mask
    inliers = [m for i, m in enumerate(matches) if mask[i]]
    outliers = [m for i, m in enumerate(matches) if not mask[i]]

    return inliers, outliers


def visualize_keypoints_matches(image1, image2, keypoints1, keypoints2, inliers, outliers):
    """
    Visualize the matches, distinguishing inliers and outliers using different colors.

    Args:
    image1 (ndarray): First image.
    image2 (ndarray): Second image.
    keypoints1 (list): List of keypoints for the first image.
    keypoints2 (list): List of keypoints for the second image.
    inliers (list): List of inlier matches.
    outliers (list): List of outlier matches.

    Displays the images with inliers in green and outliers in red.
    """
    # Draw inlier matches in green
    inlier_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inliers, None,
                                   matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Draw outlier matches in red
    outlier_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, outliers, None,
                                    matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Combine inlier and outlier images
    combined_image = cv2.addWeighted(inlier_image, 0.5, outlier_image, 0.5, 0)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.axis('off')
    plt.show()
