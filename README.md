# 2021 [한국어 질의응답 AI 경진대회](http://ai-competition.kaist.ac.kr/competitions/outline)

## 1. Train Data Download
다운받은 data는 "feature_file/video-narr/"에 위치 시켜주세요.

- [video-narr_train_questions.pt, video-narr_val_questions.pt, video-narr_vocab.json, video-narr_appearance_feat.h5 다운로드](https://drive.google.com/file/d/15dUXKfrR5eUAa2NIK_Oid6SdTcvyfzPg/view?usp=sharing)
- [video-narr_motion_feat.h5 만 다운로드](https://drive.google.com/file/d/1smHeKz-doCJbMo9gXhc6r3CsMZYpniQf/view?usp=sharing) 

## 2. feature 추출 진행

- [raw data](https://drive.google.com/file/d/1fbMB1XQvJCa2ODV0ssHYSlSXGf1fe13E/view?usp=sharing)    
- Train data의 경우 주최즉에서 제공한 [Feature data](https://drive.google.com/file/d/15dUXKfrR5eUAa2NIK_Oid6SdTcvyfzPg/view) 활용
- Test data question feature 추출
```bash
python3 preprocess/preprocess_questions.py --dataset video-narr --glove_pt 'glove.korean.pkl dir' --mode test --video_dir <test video dir> --output_pt '../feature_file/video-narr/video-narr_test_questions.pt'
```
- Test data appearance feature 추출
```bash
python3 preprocess/preprocess_features.py --gpu_id 0 --dataset video-narr --model resnet101 --video_dir <test video dir> 
```
- Test data motion feature 추출
```bash
python3 preprocess/preprocess_features.py --gpu_id 0 --dataset video-narr --model resnext101 --image_height 224 --image_width 224 --video_dir <test video dir>
```

## 3. 학습 진행

학습은 구글의 코랩 프로에서 진행했습니다.

- Baseline parameter 설정 yaml
```yaml
# configs/video_narr_v0.yaml
gpu_id: 0
multi_gpus: True
num_workers: 2
seed: 2021
exp_name: 'expVIDEO-NARR-v0'

train:
  lr: 0.0001
  batch_size: 32
  restore: False
  max_epochs: 25
  word_dim: 100 #300
  module_dim: 512
  glove: True
  k_max_frame_level: 16
  k_max_clip_level: 8
  spl_resolution: 1
  model: 'none'

val:
  flag: True

test:
  test_num: 0
  write_preds: True

dataset:
  name: 'video-narr'
  question_type: 'none'
  data_dir: '../feature_file/video-narr'
  save_dir: 'results/'
```

- Baseline 학습 진행 명령어
```bash
	python3 train.py --cfg configs/video_narr_v0.yml
```

- Attention V1 model
```python
# model/HCRM.py
class OutputUnitMultiChoicesWithAttention_v1(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoicesWithAttention_v1, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim * 2)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)
        
        self.q_attention = nn.MultiheadAttention(module_dim * 2,
                                               num_heads=1,
                                               dropout=0.15,
                                               batch_first=True)
        
        self.a_attention = nn.MultiheadAttention(module_dim * 2,
                                               num_heads=1,
                                               dropout=0.15,
                                               batch_first=True)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding,
                     ans_candidates_embedding, a_visual_embedding):
        """
        Args:
            question_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
            q_visual_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
            ans_candidates_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
            a_visual_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
        return:
            out: [Tensor] (batch_size, module_dim) => (160, 512)
        """
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)

        q_attn_embedding = torch.cat([q_visual_embedding, ans_candidates_embedding], 1)
        q_attn_output = self.q_attention(question_embedding.unsqueeze(1),
                                       q_attn_embedding.unsqueeze(1),
                                       q_attn_embedding.unsqueeze(1),
                                       need_weights=False)[0]

        a_attn_embedding = torch.cat([a_visual_embedding, ans_candidates_embedding], 1)
        a_attn_output = self.a_attention(question_embedding.unsqueeze(1),
                                       a_attn_embedding.unsqueeze(1),
                                       a_attn_embedding.unsqueeze(1),
                                       need_weights=False)[0]
        
        out = torch.cat([q_attn_output.squeeze(1), a_attn_output.squeeze(1)], 1)
        out = self.classifier(out)

        return out
```

- Attention V1 model parameter 설정 yaml
```yaml
# configs/video_narr_v1.yaml
gpu_id: 0
multi_gpus: True
num_workers: 2
seed: 2021
exp_name: 'expVIDEO-NARR-v1'

train:
  lr: 0.0001
  batch_size: 32
  restore: False
  max_epochs: 25
  word_dim: 100 #300
  module_dim: 512
  glove: True
  k_max_frame_level: 16
  k_max_clip_level: 8
  spl_resolution: 1
  model: 'attention_v1'

val:
  flag: True

test:
  test_num: 0
  write_preds: True

dataset:
  name: 'video-narr'
  question_type: 'none'
  data_dir: '../feature_file/video-narr'
  save_dir: 'results/'
```

- Attention V1 model 학습 진행 명령어
```bash
	python3 train.py --cfg configs/video_narr_v1.yml
```

- Attention V2 model
```python
# model/HCRM.py
class OutputUnitMultiChoicesWithAttention_v2(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoicesWithAttention_v2, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)
        
        self.attention = nn.MultiheadAttention(module_dim * 2,
                                               num_heads=1,
                                               dropout=0.15,
                                               batch_first=True)
        
        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding,
                     ans_candidates_embedding, a_visual_embedding):
        """
        Args:
            question_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
            q_visual_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
            ans_candidates_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
            a_visual_embedding: [Tensor] (batch_size, module_dim) => (160, 512)
        return:
            out: [Tensor] (batch_size, module_dim) => (160, 512)
        """
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)

        attn_embedding_q = torch.cat([question_embedding, ans_candidates_embedding], 1)
        attn_embedding_k = torch.cat([q_visual_embedding, a_visual_embedding], 1)
        
        attn_output = self.attention(attn_embedding_q.unsqueeze(1),
                                       attn_embedding_k.unsqueeze(1),
                                       attn_embedding_k.unsqueeze(1),
                                       need_weights=False)[0]
        
        out = self.classifier(attn_output.squeeze(1))

        return out
```

- Attention V2 model parameter 설정 yaml
```yaml
# configs/video_narr_v2.yaml
gpu_id: 0
multi_gpus: True
num_workers: 2
seed: 2021
exp_name: 'expVIDEO-NARR-v2'

train:
  lr: 0.0001
  batch_size: 32
  restore: False
  max_epochs: 25
  word_dim: 100 #300
  module_dim: 512
  glove: True
  k_max_frame_level: 16
  k_max_clip_level: 8
  spl_resolution: 1
  model: 'attention_v2'

val:
  flag: True

test:
  test_num: 0
  write_preds: True

dataset:
  name: 'video-narr'
  question_type: 'none'
  data_dir: '../feature_file/video-narr'
  save_dir: 'results/'
```

- Attention V2 model 학습 진행 명령어
```bash
	python3 train.py --cfg configs/video_narr_v2.yml
```

## 4. Submission File

- baseline
```bash
    python3 test.py --cfg configs/video_narr_v0.yml
```

- attetnion v1
```bash
    python3 test.py --cfg configs/video_narr_v1.yml
```

- attetnion v2
```bash
    python3 test.py --cfg configs/video_narr_v2.yml
```

- Ensemble
```python
import json
from scipy.stats import mode

with open("output1.json", "r") as f:
    file_1 = json.load(f)
    
with open("output2.json", "r") as f:
    file_2 = json.load(f)
    
with open("output3.json", "r") as f:
    file_3 = json.load(f)
    
result = {}
for k in file_1.keys():
    result[str(k)] = int(mode([file_1[k], file_2[k], file_3[k]]).mode[0])
    
with open('submission.json','w') as f:
    json.dump(result,f)
```
