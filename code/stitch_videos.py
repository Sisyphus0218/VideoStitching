import os
import cv2
from tqdm import tqdm
from sift import SiftDetector
from stitch_images import ImageStitcher


class VideoStitcher:
    def __init__(self):
        self.sift_detector = SiftDetector()
        self.image_stitcher = ImageStitcher()

    def read_videos(self, videos_folder):
        videos = []
        for video in os.listdir(videos_folder):
            video_path = os.path.join(videos_folder, video)
            videos.append(cv2.VideoCapture(video_path))
        return videos

    def stitch_videos(self, videos, output_path):
        # Get video information
        fps = videos[0].get(cv2.CAP_PROP_FPS)
        num_frames = int(videos[0].get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Videos Information: FPS={fps}, Total Frames={num_frames}")

        # Use the first frame of each video to compute homographies
        first_frames = []
        for video in videos:
            ret, frame = video.read()
            if ret:
                first_frames.append(frame)

        # TEST
        # self.image_stitcher.stitch_images(first_frames, "test.png")

        stitch_order = self.image_stitcher.find_stitch_order(first_frames)
        start = stitch_order[0][0]

        homographies = self.image_stitcher.compute_homographies(
            first_frames, stitch_order
        )

        output_width, output_height, transform_matrices = (
            self.image_stitcher.compute_canvas_params(first_frames, homographies, start)
        )

        weight_maps = self.image_stitcher.create_weight_maps(
            first_frames, transform_matrices, output_width, output_height
        )

        result = self.image_stitcher.warp_and_blend(
            first_frames, transform_matrices, output_width, output_height, weight_maps
        )

        height, width = result.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Video size: {width}x{height}")

        # Reset video frames to the beginning
        for video in videos:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Stitch videos frame by frame
        # pbar = tqdm(range(96)) # TEST
        pbar = tqdm(range(num_frames))
        pbar.set_description("Stitching videos...")

        for _ in pbar:
            frames = []
            for video in videos:
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) != len(videos):
                break

            result = self.image_stitcher.warp_and_blend(
                frames, transform_matrices, output_width, output_height, weight_maps
            )

            if result.shape[:2] != (height, width):
                result = cv2.resize(result, (width, height))

            out.write(result)

        # Release resources
        for video in videos:
            video.release()
        out.release()

        print(f"\nStitching completed! Video saved to {output_path}")


if __name__ == "__main__":
    videos_folder = "data/videos/"
    output_path = "result/panorama_video.mp4"

    stitcher = VideoStitcher()
    videos = stitcher.read_videos(videos_folder)
    stitcher.stitch_videos(videos, output_path)
