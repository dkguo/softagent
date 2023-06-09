import numpy as np

from moviepy.editor import *


# List of paths to your GIF files
gif_paths = [
    "./cem_sg_6/PourWaterSceneGraph_new.gif", 
    "./cem_sg_14/PourWaterSceneGraph_new.gif", 
    "./cem_sg_15/PourWaterSceneGraph_new.gif", 
    "./cem_sg_23/PourWaterSceneGraph_new.gif", 
]

# Load the GIFs using the load_gif function
gifs = []
for path in gif_paths:
    gif = VideoFileClip(path)
#    gif.preview()
#    print(gif)
    gifs.append(gif)

gifs_clip = clips_array([gifs])
#gifs_clip.preview()
    
# Combine the GIFs side by side
#clip = concatenate_videoclips([[*gifs])
    
pw_gif = VideoFileClip('./cem_pw_9/PourWater.gif')
#pw_gif.preview()

clip = clips_array([[pw_gif], [gifs_clip]]).subclip(0, 5)
#clip.preview()    

# Write the final clip to a file
clip.write_gif("combined_gifs.gif")