# Tensorflow-Computer-Vision-Models
Computer Vision models from TFHub / Swin Transformer

Tensorflow implementation of Swin Transformer used:

https://github.com/rishigami/Swin-Transformer-TF

Super Resolution:

https://www.tensorflow.org/hub/tutorials/image_enhancing


### Assignment 1
- Internet Images train set + CIFAR10 test set
- Train set scaled to 224x224. Test set upscaled via Super Resolution.
- ViT B8 + Single Dense Layer for output. No fine-tuning used.

### Assignment 2
- Fine Grained Visual Recognition Task
- Swin Transformer (swin-base-384) + Effnetv2 (efficientnetv2-s)
- Trained last block of Swin transformer
- Trained last layer of Effnetv2
