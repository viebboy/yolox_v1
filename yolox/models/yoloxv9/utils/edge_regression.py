# type: ignore
"""
From https://git.taservs.net/axon-research/vanhelsing/blob/1e8703d33040adfffaf10193aa90c04f11f94e14/vanhelsing/edge_regression.py
"""
from math import pi

import cv2
import numpy as np


def _get_transformation_matrix(
        img,
        focal,
        theta,
        phi,
        gamma,
        dx=0,
        dy=0,
        dz=0,
):
    """
    Helper function to get the transformation matrix for 3d rotation
    :param img:
    :param focal:
    :param theta:
    :param phi:
    :param gamma:
    :param dx:
    :param dy:
    :param dz:
    :return:
    """
    w = img.shape[1]
    h = img.shape[0]
    f = focal

    # Projection 2D -> 3D matrix
    A1 = np.array([
        [1, 0, -w / 2],
        [0, 1, -h / 2],
        [0, 0, 1],
        [0, 0, 1],
    ])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])

    RY = np.array([
        [np.cos(phi), 0, -np.sin(phi), 0],
        [0, 1, 0, 0],
        [np.sin(phi), 0, np.cos(phi), 0],
        [0, 0, 0, 1],
    ])

    RZ = np.array([
        [np.cos(gamma), -np.sin(gamma), 0, 0],
        [np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1],
    ])

    # Projection 3D -> 2D matrix
    A2 = np.array([
        [f, 0, w / 2, 0],
        [0, f, h / 2, 0],
        [0, 0, 1, 0],
    ])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def _deg_to_rad(deg):
    return deg * pi / 180.0


def _get_rad(theta, phi, gamma):
    return (
        _deg_to_rad(theta),
        _deg_to_rad(phi),
        _deg_to_rad(gamma),
    )


def rotate_along_axis(
    img,
    x_rotation,
    y_rotation,
    z_rotation,
    points_np,
    border_value=(
        128,
        128,
        128,
    ),
):
    """
    Rotate an image along axis.
    :param border_value: fill value of blank points
    :param points_np: coordinates of key points that will get rotated along
    :param img:
    :param x_rotation: theta, rotation of the top/bottom moving backward/forward wrt to background, along the x axis
    :param y_rotation: phi, rotation of the side moving forward and backward wrt to background, along the y axis
    :param z_rotation: gamma, normal rotation of image, along the z axis
    :return:
    """
    theta = x_rotation
    phi = y_rotation
    gamma = z_rotation
    height = img.shape[0]
    width = img.shape[1]
    # Get radius of rotation along 3 axes
    rtheta, rphi, rgamma = _get_rad(theta, phi, gamma)

    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(height ** 2 + width ** 2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # Get projection matrix
    mat = _get_transformation_matrix(
        img=img,
        focal=focal,
        theta=rtheta,
        phi=rphi,
        gamma=rgamma,
        dx=0,
        dy=0,
        dz=dz,
    )
    new_points = None
    if points_np is not None:
        new_points = cv2.perspectiveTransform(points_np, mat)

    return cv2.warpPerspective(
        img.copy(), mat, (width, height), borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    ), new_points
