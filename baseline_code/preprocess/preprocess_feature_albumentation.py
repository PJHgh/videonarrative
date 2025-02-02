preprocess_features_alb.py
액세스 권한이 있는 사용자
류
­
시스템 속성
유형
텍스트
크기
15KB
사용한 용량
15KB
위치
preprocess
소유자
나
수정 날짜
2021. 12. 15.에 내가 수정
열어 본 날짜
오후 12:43에 내가 열어 봄
생성 날짜
2021. 12. 15.에 Google Drive for desktop 사용
설명 추가
뷰어가 다운로드할 수 있음
import argparse, os
import h5py
from scipy.misc import imresize
import skvideo.io
from PIL import Image
import cv2

import torch
from torch import nn
import torchvision
import random
import numpy as np

from models import resnext
from models import densenet
from datautils import utils
from datautils import video_narr
from tqdm import tqdm
import albumentations
import albumentations.pytorch

def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model


def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=224, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('data/preprocess/pretrained/resnext-101-kinetics.pth')
    model_data = torch.load('data/preprocess/pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model


albumentations_transform = albumentations.Compose([
    albumentations.Resize(256, 256), 
    albumentations.RandomCrop(224, 224),
    albumentations.OneOf([
      albumentations.RandomGamma(gamma_limit=(80, 120), p=0.5),
      albumentations.RandomBrightness(limit=0.2, p=0.5),
      albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                      val_shift_limit=10, p=.9),
      albumentations.ShiftScaleRotate(
      shift_limit=0.0625, scale_limit=0.1, 
      rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
      albumentations.RandomContrast(limit=0.2, p=0.5)]),
    albumentations.pytorch.transforms.ToTensor()])
        


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats

# frame에서 clip feature 추출
def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    try:
        video_data = skvideo.io.vread(path) #, verbosity=1)
    except:
        print('file {} error'.format(path))
        valid = False
        if args.model == 'resnext101':
            # return list(np.zeros(shape=(num_clips, 3, num_frames_per_clip, 112, 112))), valid
            return list(np.zeros(shape=(num_clips, 3, num_frames_per_clip, 224, 224))), valid
        else:
            return list(np.zeros(shape=(num_clips, num_frames_per_clip, 3, 224, 224))), valid
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
        
            frame_data = clip[j]
            img = Image.fromarray(frame_data).convert('RGB')
            # print(f'{str(j).zfill(5)} : ','albumentations')
            # print(img)
            # print(type(img))
            img = albumentations_transform(image=np.array(img))
            # print(type(img))
            # print(img.keys())
            # print(img['image'].size())
            img = imresize(img['image'], img_size, interp='bicubic')
            # img = img['image'].transpose(2, 0, 1)[None]
            img = img.transpose(2, 0, 1)[None]
            # print(img.shape)
            # break
            frame_data = np.array(img)
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        if args.model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        # print(new_clip.shape)
        clips.append(new_clip)
    return clips, valid

# video feature 데이터를 추출하고 이를 h5 file로 생성
def generate_h5(model, video_ids, num_clips, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if args.dataset == "tgif-qa":
        if not os.path.exists('data/tgif-qa/{}'.format(args.question_type)):
            os.makedirs('data/tgif-qa/{}'.format(args.question_type))
    else:
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(video_ids)

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        video_ids_dset = None
        i0 = 0
        _t = {'misc': utils.Timer()}
        
        # 변경된 video id 처리
        for i, video_path in enumerate(tqdm(video_ids[:10000])):
            # print("preprocess_features video_path: %s" % video_path)
            text = video_path.split('/')
            sample_text= text[-1]
            # print("preprocess_features sample_text: %s" % sample_text)
            
            # 문자로 구성된 video id를 처리할 수 있도록 숫자로 변경 ( 문자 -> 숫자)
            if 'A' in sample_text:
                video_id = "1"+sample_text[11:-4]
            elif 'B' in sample_text:
                video_id = "2"+sample_text[11:-4]
            elif 'C' in sample_text:
                video_id = "3"+sample_text[11:-4]
            elif 'D' in sample_text:
                video_id = "4"+sample_text[11:-4]
            elif 'E' in sample_text:
                video_id = "5"+sample_text[11:-4]
            elif 'F' in sample_text:
                video_id = "6"+sample_text[11:-4]
            elif 'G' in sample_text:
                video_id = "7"+sample_text[11:-4]
            elif 'H' in sample_text:
                video_id = "8"+sample_text[11:-4]
            elif 'J' in sample_text:
                video_id = "9"+sample_text[11:-4]
            elif 'K' in sample_text:
                video_id = "10"+sample_text[11:-4]
            elif 'L' in sample_text:
                video_id = "11"+sample_text[11:-4]
            elif 'M' in sample_text:
                video_id = "12"+sample_text[11:-4]
            elif 'I' in sample_text:
                video_id = "13"+sample_text[11:-4]

            print("preprocess_features video_id: %s" % video_id)
            ###############################################################

            _t['misc'].tic()
            clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=16)
            if args.feature_type == 'appearance':
                clip_feat = []
                if valid:
                    for clip_id, clip in enumerate(clips):
                        feats = run_batch(clip, model)  # (16, 2048)
                        feats = feats.squeeze()
                        clip_feat.append(feats)
                else:
                    clip_feat = np.zeros(shape=(num_clips, 16, 2048))
                clip_feat = np.asarray(clip_feat)  # (8, 16, 2048)
                if feat_dset is None:
                    C, F, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int) 
            elif args.feature_type == 'motion':
                clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                if valid:
                    clip_feat = model(clip_torch)  # (8, 2048)
                    clip_feat = clip_feat.squeeze()
                    clip_feat = clip_feat.detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(num_clips, 2048))
                if feat_dset is None:
                    C, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

            i1 = i0 + 1
            feat_dset[i0:i1] = clip_feat
            video_ids_dset[i0:i1] = video_id
            i0 = i1
            _t['misc'].toc()
            if (i % 1000 == 0):
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                      .format(i1, dataset_size, _t['misc'].average_time,
                              _t['misc'].average_time * (dataset_size - i1) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=2, help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msvd-qa', 'msrvtt-qa','video-narr'], type=str)
    parser.add_argument('--question_type', default='none', choices=['frameqa', 'count', 'transition', 'action', 'none'], type=str)
    parser.add_argument('--video_dir', default='/data/dekim/video-qa/data', help='base directory of data')

    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="data/{}/{}_{}_feat_TTA.h5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=8, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--model', default='resnet101', choices=['resnet101', 'resnext101'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')

    args = parser.parse_args()
    if args.model == 'resnet101':
        args.feature_type = 'appearance'
    elif args.model == 'resnext101':
        args.feature_type = 'motion'
    else:
        raise Exception('Feature type not supported!')
    # set gpu
    if args.model != 'resnext101':
        torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # annotation files - 본 경진대회는 video-narr dataset 사용.
    if args.dataset == 'tgif-qa':

        args.annotation_file = '/home/tgif-qa-master/dataset/Total_{}_question.csv'
        args.video_dir = '/data/TGIF/gifs'
        '''
        args.annotation_file = '/ceph-g/lethao/datasets/tgif-qa/csv/Total_{}_question.csv'
        args.video_dir = '/ceph-g/lethao/datasets/tgif-qa/gifs' 
        '''
        args.outfile = 'data/{}/{}/{}_{}_{}_feat.h5'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.question_type, args.dataset, args.question_type, args.feature_type))
    elif args.dataset == 'msrvtt-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msrvtt/annotations/{}_qa.json'
        args.video_dir = '/ceph-g/lethao/datasets/msrvtt/videos/'
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))

    elif args.dataset == 'msvd-qa':
        args.annotation_file = '/ceph-g/lethao/datasets/msvd/MSVD-QA/{}_qa.json'
        args.video_dir = '/ceph-g/lethao/datasets/msvd/MSVD-QA/video/'
        args.video_name_mapping = '/ceph-g/lethao/datasets/msvd/youtube_mapping.txt'
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.dataset, args.feature_type))
    
    elif args.dataset == 'video-narr':
        video_paths = video_narr.load_video_paths(args)
        random.shuffle(video_paths)
        print("Number of unique videos: {}", format(len(video_paths)))

        # load model
        if args.model == 'resnet101':
            model = build_resnet()
        elif args.model == 'resnext101':
            model = build_resnext()
        generate_h5(model, video_paths, args.num_clips, args.outfile.format(args.dataset, args.dataset, args.feature_type))#model, video_ids, num_clips, outfile