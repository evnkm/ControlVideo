import cv2
def video_to_images(input_vid):
    video = cv2.VideoCapture(input_vid)

    images = []
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # if count % 10 == 0:
        images.append(frame)
        count += 1
    video.release()

    return images

if __name__ == "__main__":
    print(len(video_to_images("data/EI_Stairs_2_with_background.mp4")))