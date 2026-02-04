import numpy as np
from scipy.spatial import distance


def eye_aspect_ratio(eye_points):
    # vertical distances
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    # horizontal distance
    C = distance.euclidean(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[2], mouth_points[3])
    B = distance.euclidean(mouth_points[4], mouth_points[5])
    C = distance.euclidean(mouth_points[0], mouth_points[1])

    mar = (A + B) / (2.0 * C)
    return mar
