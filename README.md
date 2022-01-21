# Face Detector
A face detector program utilizing MTCNN architecture

We use pretrained model weights from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) re-trained on WIDER face dataset and our own face dataset (custom_dataset folder).

Training:
1. extract WIDER train dataset on build_dataset folder
2. extract annotations from WIDER face on build_dataset folder
3. merge custom dataset images & annotations
4. run trainer.py to start training (trained weights will be saved in train folder)

Eval:
import the MTCNN class from mtcnn.py then feed any face images to start inferring (done in exp.ipynb)