import cv2
import os

def extract_frames(video_path,
                   output_folder):

    os.makedirs(output_folder,
                exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(
            output_folder,
            f"frame_{count:04d}.png"
        )

        cv2.imwrite(frame_path,
                    frame)

        count += 1

    cap.release()

    return fps