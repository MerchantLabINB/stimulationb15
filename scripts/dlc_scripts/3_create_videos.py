import deeplabcut

config_path = r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\config.yaml"
deeplabcut.create_labeled_video(
    config_path, 
    videos=[r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos"], 
    videotype='.mp4', 
    shuffle=2, 
    filtered=True, 
    trailpoints=7, 
    save_frames=True, 
    draw_skeleton=True, 
    overwrite=True,
    pcutoff=0.6,
    dotsize=1,
    fastmode=False
    #destfolder=r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos"
)