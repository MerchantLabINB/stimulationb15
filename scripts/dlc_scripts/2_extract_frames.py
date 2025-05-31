import deeplabcut
            
#help(deeplabcut.extract_frames)
config_path = r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\config.yaml"
random_20_videos = [r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240528_191800665_3727_37270.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240509_192829762_3588_35880.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240518_153104163_4500_450.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240515_183853353_12856_128560.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240528_191800665_2017_20170.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240528_202051471_6183_61829.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240524_191025834_5240_52399.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240524_201355224_3580_35799.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240509_191526862_14906_14960.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240509_191905880_3654_36539.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240515_185600664_5771_57710.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240524_190146903_7631_76310.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240524_194108473_3391_33909.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240528_201847508_1103_1129.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240528_195931576_5592_55920.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240523_193310834_3445_34450.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240523_194358295_4010_40100.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240518_152236707_1140_11400.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240515_190212033_6661_66609.mp4",
r"C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos\40298451__20240523_193310834_8388_83879.mp4"]


deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False, videos_list = random_20_videos)

