# Introduction
This is Regularizing Visual Semantic Embedding with Contrastive Learning for Image-Text Matching, source code of [ConVSE](https://ieeexplore.ieee.org/abstract/document/9785732). This paper accepted by IEEE SPL. It is built on the top of the [VSE$\infty$](https://github.com/woodfrog/vse_infty/tree/bigru) in PyTorch.
# Requirements and Installation
We recommended the following dependencies.
- Python3.6+
- Pytorch 1.9.0+

# Download data
Download the dataset files. We use the image feature created by SCAN, download here[https://github.com/kuanghuei/SCAN].

# Training new models
Run `train.py`:
```
python train.py --data_path "$DATA_PATH" --data_name "$DATA_NAME" --vocab_paath "$VOCAB_PATH" --model_name "runs/convse/model/" --use_contrastive
```

# Evaluate trained models 
```Python
from vocab import Vocabulary
import evalution
evalution.evalrank("$PATH/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```

# Reference
If you found this code useful, please cite the following paper:
```
@article{liu2022regularizing,
  title={Regularizing Visual Semantic Embedding with Contrastive Learning for Image-Text Matching},
  author={Liu, Yang and Liu, Hong and Wang, Huaqiu and Liu, Mengyuan},
  journal={IEEE Signal Processing Letters},
  year={2022},
  publisher={IEEE}
}
```
