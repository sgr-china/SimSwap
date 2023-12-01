import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import torch.nn.functional as F
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = True,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    # mse = torch.nn.MSELoss().cuda()
    mse = torch.nn.MSELoss()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        # net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            # 使用人脸检测模型对当前帧进行人脸检测
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                "人脸图像列表"
                frame_align_crop_list = detect_results[0]

                "人脸对应的仿射矩阵列表"
                frame_mat_list = detect_results[1]
                "用于存储不同人脸与目标身份向量相似度的列表"
                id_compare_values = [] 
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:

                    # frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                    "将裁剪后的人脸图像转换为 PyTorch 张量"
                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))[None, ...]
                    "对裁剪后的人脸进行特定规范化"
                    frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                    "对特定规范化后的人脸图像进行下采样"
                    frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                    "使用 ArcFace 模型获取裁剪后的人脸的身份特征"
                    frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)
                    "计算当前人脸与目标身份向量的相似度，并将结果添加到列表中"
                    id_compare_values.append(mse(frame_align_crop_crop_id_nonorm, specific_person_id_nonorm).detach().cpu().numpy())
                    "将当前裁剪后的人脸图像添加到列表中"
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                id_compare_values_array = np.array(id_compare_values)
                min_index = np.argmin(id_compare_values_array)
                min_value = id_compare_values_array[min_index]
                if min_value < id_thres:
                    swap_result = swap_model(None, frame_align_crop_tenor_list[min_index], id_vetor, None, True)[0]
                
                    reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,\
                        os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask= use_mask, norm = spNorm)
                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
        else:
            break

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')

