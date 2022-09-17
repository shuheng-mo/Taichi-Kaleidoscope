from moviepy.editor import VideoFileClip

videoClip = VideoFileClip("videos/sand_piles.avi")
videoClip.write_gif("sand_piles.gif")

