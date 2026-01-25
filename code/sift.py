import cv2


class SiftDetector:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, image):
        image_blurred = cv2.GaussianBlur(image, (3, 3), 0)
        keypoints, descriptors = self.sift.detectAndCompute(image_blurred, None)
        return keypoints, descriptors

    def match_images(self, image1, image2):
        keypoints1, descriptors1 = self.detect_and_compute(image1)
        keypoints2, descriptors2 = self.detect_and_compute(image2)

        bf_matcher = cv2.BFMatcher()
        knn_matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

        matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                matches.append(m)

        # visualize matches
        # match_result_image = cv2.drawMatches(
        #     image1,
        #     keypoints1,
        #     image2,
        #     keypoints2,
        #     matches,
        #     None,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        # )
        # cv2.imshow("Match Result", match_result_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return matches


if __name__ == "__main__":
    image1_path = "data/images/test_image1.png"
    image2_path = "data/images/test_image2.png"

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    sift_detector = SiftDetector()
    matches = sift_detector.match_images(image1, image2)
