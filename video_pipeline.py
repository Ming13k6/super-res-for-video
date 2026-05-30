from extract_frames import extract_frames

video_path = "input/test.mp4"

fps = extract_frames(
    video_path,
    "video_frames"
)

print("Frames extracted.")
print(f"FPS: {fps}")

print("Now run:")
print("python video_inference.py")