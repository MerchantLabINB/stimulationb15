from moviepy.editor import VideoFileClip

# Create a VideoFileClip object
clip = VideoFileClip('output.mp4')

# Get video information
duration = clip.duration  # duration in seconds
frame_rate = clip.fps  # frames per second
size = clip.size  # size in pixels (width, height)

# Print video information
print(f"Duration: {duration} seconds")
print(f"Frame rate: {frame_rate} fps")
print(f"Size: {size[0]} x {size[1]} pixels")