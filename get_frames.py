import os
import cv2
import random

# Returns a list of tuples, first element of which is abspath of video file, second
# element is list of frame numbers
def get_random_frames():
	TOTAL_FRAMES = 350
	basepath = './datasets/UCF-101'

	action_names = os.listdir(basepath)
	num_actions = len(action_names)

	frames_per_action = TOTAL_FRAMES // num_actions

	final = []

	for action in action_names:
		folder_path = os.path.join(basepath, action)
		all_videos = os.listdir(folder_path)
		#num_clips = max([int(s[s.index('g') + 1 : s.index('g') + 3]) for s in all_videos])
		num_clips = len(all_videos)

		frames_per_clip = frames_per_action // num_clips

		for video_name in all_videos:
			filename = os.path.join(folder_path, video_name)
			vidcap = cv2.VideoCapture(filename)
			length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
			frame_numbers = random.sample(range(1, length), frames_per_clip)
			final.append((os.path.abspath(filename), frame_numbers))

	return final
