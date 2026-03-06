from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen3VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

from torch.nn import L1Loss, NLLLoss
from mmengine.visualization import Visualizer, TensorboardVisBackend
from projects.ScanVLA_RefCOCOGaze_Ablation.models.mllm.qwen3vl import Qwen3VL

from projects.ScanVLA_RefCOCOGaze_Ablation.models import ScanVLAModel, DirectResize, InternVLMLLM
from projects.ScanVLA_RefCOCOGaze_Ablation.datasets import ReferGazeDataset, scanvla_collect_fn

from projects.ScanVLA_RefCOCOGaze_Ablation.models.ScanVLA_decoder import TransformerDecoderWrapper

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = 'pretrained/InternVL3-2B'
# pretrained_pth = None
pretrained_pth = 'pretrained/model_internvl3_2b.pth'

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
# batch_size = 8  # per_device
batch_size = 1  # per_device

accumulative_counts = 64
dataloader_num_workers = 64

# accumulative_counts = 1
# dataloader_num_workers = 1

max_epochs = 6

optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 4  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=ScanVLAModel,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    # arch_type='qwen',
    decoder=dict(
        type=TransformerDecoderWrapper,
        activation="relu",
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout_attn = 0.1,
        dropout_mlp = 0.15,
        num_decoder_layers=6,
        max_len=4,
        input_dim=1536,
        args=None,
    ),
    mllm=dict(
        type=InternVLMLLM,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=["embed_tokens", "lm_head"]
        ),
    ),
    tokenizer=tokenizer,
    loss_fn_xy = dict(
        type=L1Loss,
        reduction='none',
        ),
    loss_fn_token = dict(
        type=NLLLoss,
        reduction='mean',
        ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT_Gaze = '/data/lyt/01-Datasets/01-ScanPath-Datasets/ART_data/data/'
train_refgazes = 'refcocogaze_train_correct_tf_512X320_6.json'
img_dir = 'images_512X320'

scanvla_default_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
)

refcoco_gaze_dataset=dict(
    type=ReferGazeDataset,
    name='RefCOCOGaze', #1
    fixs= DATA_ROOT_Gaze + train_refgazes,
    img_dir= DATA_ROOT_Gaze + img_dir,
    num_classes_per_sample=1, #5
    repeats=1, #5
    **scanvla_default_dataset_configs
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=refcoco_gaze_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=scanvla_collect_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
# visualizer = None
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)