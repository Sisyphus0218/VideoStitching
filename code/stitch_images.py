import os
import cv2
import numpy as np
from sift import SiftDetector


class ImageStitcher:
    def __init__(self):
        self.sift_detector = SiftDetector()

    def read_images(self, images_folder):
        """
        Read images from given folder.

        :param images_folder: Folder containing images
        :return images: List of images
        """
        images = []
        for image in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image)
            images.append(cv2.imread(image_path))
        return images

    def build_match_graph(self, images):
        """
        Compute the number of matching points between each pair of images.

        :param images: List of images
        :return match_matrix: Matrix of number of matches between each pair of images
        """
        num_images = len(images)

        # match_matrix[i][j]: number of matches between image i and j
        match_matrix = np.zeros((num_images, num_images), dtype=int)

        for i in range(num_images):
            for j in range(i + 1, num_images):
                num_matches = self.sift_detector.match_images(images[i], images[j])
                match_matrix[i][j] = len(num_matches)
                match_matrix[j][i] = len(num_matches)

        # INFO
        # print("Image matching result:")
        # for i in range(num_images):
        #     for j in range(i + 1, num_images):
        #         num_matches = match_matrix[i][j]
        #         print(f"Image {i+1} <---> Image {j+1}: {num_matches} matching points")

        return match_matrix

    def find_stitch_order(self, images):
        """
        Find the order of stitching images.

        :param images: List of images
        :return stitch_order: List of (target_image_index, source_image_index)
        """
        # match_matrix[i][j]: number of matches between image i and j
        match_matrix = self.build_match_graph(images)

        # start from the image with most matches
        num_matches_sum = np.sum(match_matrix, axis=1)
        start = int(np.argmax(num_matches_sum))

        # stitch_order: list of (target_image_index, source_image_index)
        stitch_order = []

        # Prim's algorithm to find the order of stitching
        num_images = len(images)
        selected = set([start])
        not_selected = set(range(0, num_images))
        not_selected.remove(start)

        while not_selected:
            max_num_matches = -1
            target_image_index = -1
            source_image_index = -1
            for i in not_selected:
                for j in selected:
                    matches = match_matrix[i][j]
                    if matches > max_num_matches:
                        max_num_matches = matches
                        target_image_index = j
                        source_image_index = i

            selected.add(source_image_index)
            not_selected.remove(source_image_index)
            stitch_order.append((target_image_index, source_image_index))

        return stitch_order

    def compute_pairwise_homography(self, image1, image2):
        """
        Compute the homography matrix from image2 to image1.

        :param image1: First image
        :param image2: Second image
        :return H: Homography matrix from image2 to image1
        """
        keypoints1, descriptors1 = self.sift_detector.detect_and_compute(image1)
        keypoints2, descriptors2 = self.sift_detector.detect_and_compute(image2)
        matches = self.sift_detector.match_images(image1, image2)

        if len(matches) < 4:
            print("Error: Not enough matches to compute homography.")
            return None

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 4.0)

        return H

    def compute_homographies(self, images, stitch_order):
        """
        Compute the homography of each image to the start image.

        :param images: List of images
        :param stitch_order: List of (target_image_index, source_image_index)
        :param start: Index of the start image
        :return homographies: List of homography matrices
        """
        start = stitch_order[0][0]
        homographies = [0] * len(images)  # homographies = [None, H1, H2, ...]
        homographies[start] = None

        for target, source in stitch_order:
            H = self.compute_pairwise_homography(images[target], images[source])
            if target != start:
                H = homographies[target] @ H
            if H is None:
                print(f"Error: image {source+1} matching failed, using identity matrix")
                H = np.eye(3)
            homographies[source] = H

        return homographies

    def compute_canvas_params(self, images, homographies, start):
        """
        Compute canvas parameters for stitching.

        :param images: List of images
        :param homographies: List of homography matrices
        :param start: Index of the start image
        :return output_width: Width of the output canvas
        :return output_height: Height of the output canvas
        :return transform_matrices: List of transformation matrices for each image
        """
        # Get the corners of all transformed images
        all_corners = []
        for i, image in enumerate(images):
            h, w = image.shape[:2]
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            if i == start:
                all_corners.append(corners)
            else:
                corners_transformed = cv2.perspectiveTransform(corners, homographies[i])
                all_corners.append(corners_transformed)

        # Combine all corners
        all_corners = np.concatenate(all_corners, axis=0)

        # Compute bounding box
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Create translation matrix to shift all content to positive coordinates
        translation = np.array(
            [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32
        )

        # Compute canvas size
        output_width = x_max - x_min
        output_height = y_max - y_min

        # Compute final transformation matrices for each image
        transform_matrices = []
        for i in range(len(images)):
            if i == start:
                transform_matrices.append(translation)
            else:
                transform_matrices.append(translation @ homographies[i])

        return output_width, output_height, transform_matrices

    def create_weight_maps(
        self, images, transform_matrices, output_width, output_height
    ):
        """
        Compute weight maps for each image based on distance transform.

        :param images: List of images
        :param transform_matrices: List of transformation matrices for each image
        :param output_width: Width of the output canvas
        :param output_height: Height of the output canvas
        :return weight_maps: List of weight maps for each image
        """
        weight_maps = []

        for i, image in enumerate(images):
            warped = cv2.warpPerspective(
                image, transform_matrices[i], (output_width, output_height)
            )
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255

            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

            if dist.max() > 0:
                dist_norm = dist / dist.max()
                weight_map = dist_norm**3
            else:
                weight_map = dist

            weight_maps.append(weight_map)

        return weight_maps

    def warp_and_blend(
        self, images, transform_matrices, output_width, output_height, weight_maps=None
    ):
        """
        Warp and blend all images into a single panorama.

        :param images: List of images
        :param transform_matrices: List of transformation matrices for each image
        :param output_width: Width of the output canvas
        :param output_height: Height of the output canvas
        :param weight_maps: Pre-computed weight maps (optional, for video stitching)
        :return result: Stitched panorama image
        """
        result = np.zeros((output_height, output_width, 3), dtype=np.float64)
        weight_sum = np.zeros((output_height, output_width), dtype=np.float64)

        if weight_maps is None:
            weight_maps = self.create_weight_maps(
                images, transform_matrices, output_width, output_height
            )

        for i, image in enumerate(images):
            warped = cv2.warpPerspective(
                image, transform_matrices[i], (output_width, output_height)
            )

            weight_map = weight_maps[i]
            weight_map_3ch = np.repeat(weight_map[:, :, np.newaxis], 3, axis=2)

            result += warped.astype(np.float64) * weight_map_3ch
            weight_sum += weight_map

        weight_sum = np.maximum(weight_sum, 1e-10)
        weight_sum_3ch = np.repeat(weight_sum[:, :, np.newaxis], 3, axis=2)

        result = (result / weight_sum_3ch).astype(np.uint8)

        return result

    def stitch_images(self, images, output_path):
        """
        Stitch images from the given folder and save the result to output path.

        :param images: List of images
        :param output_path: Output image path
        :return result: Stitched image
        """
        # Determine stitch order and the start image
        stitch_order = self.find_stitch_order(images)
        start = stitch_order[0][0]

        # Compute the homography of each image to the start image
        homographies = self.compute_homographies(images, stitch_order)

        # Compute canvas parameters
        output_width, output_height, transform_matrices = self.compute_canvas_params(
            images, homographies, start
        )

        result = self.warp_and_blend(
            images, transform_matrices, output_width, output_height
        )

        # Save result
        cv2.imwrite(output_path, result)

        return result


if __name__ == "__main__":
    images_folder = "data/images/"
    output_path = "result/panorama_image.png"

    stitcher = ImageStitcher()
    images = stitcher.read_images(images_folder)
    stitcher.stitch_images(images, output_path)

    print(f"Stitching completed! Image saved to {output_path}")
