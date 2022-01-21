# Face Detector
A face detector program utilizing MTCNN architecture

We use pretrained model weights from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) re-trained on WIDER face dataset and our own face dataset (custom_dataset folder).

## Dataset augmentation
1. download WIDER train dataset & annotations from [WIDER Face](http://shuoyang1213.me/WIDERFACE/)
2. extract WIDER train dataset on build_dataset folder
3. extract annotations from WIDER face on build_dataset folder
4. merge custom dataset images & annotations
5. run generate_data.py to generate 12x12, 24x24, 48x48 faces & non-faces images

## Training
run trainer.py to start training (trained weights will be saved in train folder)

## Eval
import the MTCNN class from mtcnn.py then feed any face images to start inferring (done in exp.ipynb)
