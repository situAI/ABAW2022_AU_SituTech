import imageio

video_path = r''
vid = 0
try:
    vid = imageio.get_reader(video_path, 'ffmpeg')
except Exception as e:
    print('video broke')
    break

vid_meta = vid.get_meta_data()

fps = vid_meta['fps']
total_duration = vid_meta['duration']
total_frame_num = fps * total_duration

for frame_index, frame in enumerate(vid):
    print(frame.shape)
