# Unified Multimodal Scanpath Prediction with Perception Enhanced Vision-Language Models
Modeling human visual attention through dynamic scanpath prediction is crucial for understanding gaze behavior and has broad applications in autonomous driving, robotics, and virtual reality. Recent multimodal scanpath prediction methods typically focus on specific task settings and often suffer from visual–semantic misalignment and limited positional awareness, restricting their generalization in real-world scenarios.
To address these limitations, we propose UniScanVLA, a unified vision–language framework for multimodal scanpath prediction. Our model supports diverse guidance settings, including target-category guidance, visual question answering, referring expression guidance, and long-text description guidance within a single architecture. By leveraging a vision–language backbone, UniScanVLA effectively alleviates visual–semantic alignment issues.
Moreover, we introduce a perception-enhanced scanpath decoder and a fixed segmentation LoRA module to improve target understanding and spatial awareness. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple multimodal scenarios.

# Environment
the same ans [Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos](https://github.com/bytedance/Sa2VA/tree/main)

or you can 
```bash
conda create -n vlm python=3.10.16
conda activate vlm
pip install -r requirements.txt
```

# Files to download
Currently, our model only supports Qwen3-VL-2B, and it is first necessary to download [OpenGVLab/InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) from Hugging Face.

then, you can use [this script](tools/convert_to_pth.py) to transfer the [segmentation lora](https://huggingface.co/ByteDance/Sa2VA-InternVL3-2B) from huggingface file to .pth format.  or you can directly download the segmentation lora from [this link](https://drive.google.com/file/d/1-nzlFI-4cDkQRgbodIXguKLFb3-YEF7c/view?usp=drive_link)

For the four scenarios of Referential Expression, Image Caption, Object Category, and Visual Question Answering (VQA)-Guided scenario, you can directly download the pre-trained model weights via [this link](https://drive.google.com/drive/folders/1bIfyqbADC__bJ04W2jvsIzeTUzwV0CuN?usp=drive_link).

For the scenario of Zero-Shot Object Category Guided Scanpath Prediction, A cross-validation method is used to calculate the final metrics, so is contains 18 model weights. model weights are comming on the soon.

At this point the project root should look like:
ScanVLA/
├── pretrained/
│     └── model_qwen_2b.pth
|     └── checkpoints/
|     └── Qwen3-VL-2B-Instruct/
└── projects/
└── test_evaluation_metrics/
└── tools/
└── vlm/

# Test
test script under test_evaluation_metrics/test_metrics_AiR.py
```bash
python test_evaluation_metrics/test_metrics_refcocogaze.py
python test_evaluation_metrics/test_metrics_LN.py
python test_evaluation_metrics/test_metrics_COCOSearch18.py
python test_evaluation_metrics/test_metrics_COCOSearch18_ZeroGaze.py
python test_evaluation_metrics/test_metrics_AiR.py
```

# Train 
you can use the following script to train, for example:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --node_rank=0 --master_addr=127.0.0.7 --master_port=29407 --nproc_per_node=2 train.py projects/ScanVLA_RefCOCOGaze/configs/ScanVLA_RefCOCOGaze.py --launcher pytorch --deepspeed deepspeed_zero2
```
Don't forget to change the correct train config file of different scenario.

if you want to debug, you shoule change the correct config file in (./vscode/launch.json), and use train.py to debug.

# Citation
This work is currently unfinished and should not be distributed or circulated!


