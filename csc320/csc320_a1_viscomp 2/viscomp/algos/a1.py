# CSC320 Fall 2022
# Assignment 1
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import numpy as np
import cv2
import viscomp.ops.image as img_ops
import pdb


def run_a1_algo(source_image, destination_image, source_coords, destination_coords, homography=None):
    """Run the entire A1 algorithm.

    Args: 
        source_image (np.ndarray): The source image of shape [Hs, Ws, 4]
        destination_image (np.ndarray): The destination image of shape [Hd, Wd, 4]
        source_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the source image.
        destination_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the destination image.
        homography (np.ndarray): (Optional) [3, 3] homography matrix. If passed in, will use this
                                 instead of calculating it.
    
    Returns:
        (np.ndarray): Written out image of shape [Hd, Wd, 4]
    """
    if homography is None:
        print("Calculating homography...")
        np.set_printoptions(formatter={'float': '{:.4f}'.format})
        homography = calculate_homography(destination_coords, source_coords)
    else:
        print("Using preset homography matrix...")
    print("")
    print("Homography matrix:")
    print(homography)
    print("")
    print("Performing backward mapping...")
    output_buffer = backward_mapping(homography, source_image, destination_image, destination_coords)
    print("Algorithm has succesfully finished running!")

    # points = homography @ np.concatenate([destination_coords, np.ones([4, 1])], axis=-1).T
    # backprojected_coords = (points.T)[:4, :2] / (points.T)[:4, 2:]
    # print(source_coords)
    # print(destination_coords)
    # print(backprojected_coords)
    return output_buffer


def convex_polygon(poly_coords, image_coords):
    """From coords that define a convex hull, find which image coordinates are inside the hull.

     Args:
         poly_coords (np.ndarray): [N, 2] list of 2D coordinates that define a convex polygon.
                              Each nth index point is connected to the (n-1)th and (n+1)th 
                              point, and the connectivity wraps around (i.e. the first and last
                              points are connected to each other)
         image_coords (np.ndarray): [H, W, 2] array of coordinates on the image. Using this,
                                 the goal is to find which of these coordinates are inside
                                 the convex hull of the polygon.
         Returns:
             (np.ndarray): [H, W] boolean mask where True means the coords is inside the hull.
     """
    mask = np.ones_like(image_coords[..., 0]).astype(np.bool)
    N = poly_coords.shape[0]
    for i in range(N):
        dv = poly_coords[(i + 1) % N] - poly_coords[i]
        winding = (image_coords - poly_coords[i][None]) * (np.flip(dv[None], axis=-1))
        winding = winding[..., 0] - winding[..., 1]
        mask = np.logical_and(mask, (winding > 0))
    return mask


# student_implementation

def calculate_homography(destination, source):
    """Calculate the homography matrix based on source and desination coordinates.

    Args:
     source (np.ndarray): [4, 2] matrix of 2D coordinates in、 the source image.
     destination (np.ndarray): [4, 2] matrix of 2D coordinates in the destination image.

    Returns:
     (np.ndarray): [3, 3] homography matrix.
    """
    x1, y1 = source[0][0], source[0][1]
    x1_d, y1_d = destination[0][0], destination[0][1]
    x2, y2 = source[1][0], source[1][1]
    x2_d, y2_d = destination[1][0], destination[1][1]
    x3, y3 = source[2][0], source[2][1]
    x3_d, y3_d = destination[2][0], destination[2][1]
    x4, y4 = source[3][0], source[3][1]
    x4_d, y4_d = destination[3][0], destination[3][1]
    # print(x1, y1, x1_d, y1_d)

    m = np.array([[x1, y1, 1, 0, 0, 0, -x1 * x1_d, -x1_d * y1],
                  [0, 0, 0, x1, y1, 1, -x1 * y1_d, -y1_d * y1],
                  [x2, y2, 1, 0, 0, 0, -x2 * x2_d, -x2_d * y2],
                  [0, 0, 0, x2, y2, 1, -x2 * y2_d, -y2_d * y2],
                  [x3, y3, 1, 0, 0, 0, -x3 * x3_d, -x3_d * y3],
                  [0, 0, 0, x3, y3, 1, -x3 * y3_d, -y3_d * y3],
                  [x4, y4, 1, 0, 0, 0, -x4 * x4_d, -x4_d * y4],
                  [0, 0, 0, x4, y4, 1, -x4 * y4_d, -y4_d * y4]])
    n = np.array([x1_d, y1_d, x2_d, y2_d, x3_d, y3_d, x4_d, y4_d])
    solution = np.linalg.solve(m, n)
    homography = np.append(solution, 1).reshape((3, 3))
    return homography


def denormalized(coords, h, w):
    coords_ = coords.copy().astype(np.float32)
    z = coords_[2][0]
    x = coords_[0][0] / z
    y = coords_[1][0] / z

    x = int(np.round((x + 1) / 2 * w))
    y = int(np.round((y + 1) / 2 * h))
    y = -y

    return x, y


def backward_mapping(transform, source_image, destination_image, destination_coords):
    """Perform backward mapping onto the destination image.

    The goal of this function is to map each destination image pixel which is within the polygon defined
    by destination_coords to a corresponding image pixel in source_image.

    Hints: Start by iterating through the destination image pixels using a nested for loop. For each pixel,
    use the convex_polygon function to find whether they are inside the polygon. If they are, figure out
    how to use the homography matrix to find the corresponding pixel in source_image.

    Args:
     transform (np.ndarray): [3, 3] homogeneous transformation matrix.
     source_image (np.ndarray): The source image of shape [Hs, Ws, 4]
     destination_image (np.ndarray): The destination image of shape [Hd, Wd, 4]
     source_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the source image.
     destination_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the destination image.

    Returns:
     (np.ndarray): [Hd, Wd, 4] image with the source image projected onto the destination image.
    """
    output_buffer = np.zeros(destination_image.shape)
    H = destination_image.shape[0]
    W = destination_image.shape[1]
    image_coords = img_ops.create_coordinates(H, W)
    masked_destination_image = convex_polygon(destination_coords, image_coords)

    for y in range(H):
        for x in range(W):
            if masked_destination_image[y][x]:
                t = np.array([np.append(image_coords[y][x], [1])]).transpose()
                source_point = np.matmul(np.linalg.inv(transform), t)
                source_pixels = denormalized(source_point, H, W)
                rgb = source_image[source_pixels[1]][source_pixels[0]]
                output_buffer[y][x] = rgb


    return output_buffer
