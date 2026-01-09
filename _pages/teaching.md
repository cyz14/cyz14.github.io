---
layout: page
permalink: /teaching/
title: AI Courses
description: AI From Scratch, Image Classification, Language Models, and Generative Models
nav: true
nav_order: 6
---

<!-- For now, this page is assumed to be a static description of your courses. You can convert it to a collection similar to `_projects/` so that you can have a dedicated page for each course.

Organize your courses by years, topics, or universities, however you like! -->

让大家入门如何上手做AI项目。主题包括AI system，图像分类模型，语言模型，生成模型。

参考课程包括：

- [Microsoft AI System course on Github](https://microsoft.github.io/AI-System/)
- [Stanford CS231n Spring 2025](https://cs231n.stanford.edu/schedule.html)
- Stony Brook CSE 538 NLP 2025 Spring
- [李宏毅 Machine Learning and Having It Deep and Structured, 2018 Spring](https://speech.ee.ntu.edu.tw/~hylee/mlds/2018-spring.php)
- [KAIST 492(D) Diffusion Models and Their Applications,24 Fall](https://mhsung.github.io/kaist-cs492d-fall-2024/)
- [MIT Diffusion course 2025](https://diffusion.csail.mit.edu/2025/index.html)
- [MIT EECS 6.7960 Deep Learning Fall 2025](https://deeplearning6-7960.github.io/)
- [MIT EECS 6.S978 Deep Generative Models Fall 2024](https://mit-6s978.github.io/)

## 1. AI system与图像分类模型

运行环境：本地nvidia gpu，或colab

目标：

- 环境搭建:gpu，anaconda/miniconda，create env, run jupyter
- ai框架（pytorch）简介
- maximum entropy classifier 逻辑回归模型
- 使用pytorch实现逻辑回归模型
- 模型训练，评测模型，观察指标

第一讲：AI system 1-4：

- 课程介绍
- 人工智能系统概述
- 计算框架基础
- 矩阵计算与体系结构
- 环境搭建:
  - 安装git
  - 安装python环境：miniconda，创建env
  - 安装jupyter notebook（打开colab试用）
  - 安装CUDA环境
    - 安装gpu driver: nvidia-smi
    - 安装cuda toolkit: nvcc compiler
    - 安装cuDNN
  - 安装pytorch，验证gpu available
  - 安装docker

练习

- [知乎专栏：安装GPU驱动，CUDA和cuDNN](https://zhuanlan.zhihu.com/p/143429249)
- cs231n python/numpy Review [[Colab]](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb), [Tutorial](https://cs231n.github.io/python-numpy-tutorial/)

第二讲：

- cs231n Lecture 1: Deep Learning for Computer Vision
- CSE 538 maximum entropy classifier (Logistic regression)
- pytorch.org tutorial with examples [tensor, function, auto grad, module](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

练习

- CSE538 Spring 2025: [Demo of Logistic Regression with Gradient Descent](https://adithya8.github.io/assets/cse538-sp25/intro_numpy_LogisticRegression.txt)
- MIT EECS 6.7960 Deep Learning Fall 2025, pytorch tutorial [[colab]](https://colab.research.google.com/drive/1nZg9_wYpVYWS9xZAiSft5_gyluuQpBWY?usp=sharing)
- CSE538 Spring 2025 Assignment 1: Multi-Class Logistic Regression [[Google Drive Download]](https://drive.google.com/drive/folders/1Uj1li3QE-EEl3SVNipbaMqsc6mGWp9_F?usp=drive_link)

第三讲：

- cs231n Lecture 2: Image Classification with Linear Classifiers
- cs231n Lecture 3: Regularization and Optimization

练习

- AI system [Lab1](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab1/README.md)： mnist，模型可视化，逐layer性能profiling
- pytorch nn tutorial [What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html): mnist 图片识别
- UvA DL Notebooks [Tutorial 2: Introduction to PyTorch](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html)

第四讲：

- cs231n Lecture 4: Neural Networks and Backpropagation
- cs231n Lecture 5: Image Classification with CNNs
- cs231n Lecture 6: Training CNNs and CNN Architectures

阅读：

- Kaiming He CVPR25 talk: [Workshop: What's After Diffusion?](https://people.csail.mit.edu/kaiming/cvpr25talk/cvpr2025_meanflow_kaiming.pdf)
- cs231n [Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [VGGNet](https://arxiv.org/abs/1409.1556)
- [GoogLeNet](https://arxiv.org/abs/1409.4842)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [tensorflow从0到N：反向传播的推导](https://github.com/EthanYuan/TensorFlow-Zero-to-N/blob/master/TensorFlow%E4%BB%8E0%E5%88%B0N/TensorFlow%E4%BB%8E0%E5%88%B01/10-NN%E5%9F%BA%E6%9C%AC%E5%8A%9F%EF%BC%9A%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%9A%84%E6%8E%A8%E5%AF%BC.md)
- shervine blog: [Cheatsheet CNN](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- shervine blog: [图片识别模型的演化](https://stanford.edu/~shervine/blog/evolution-image-classification-explained)：LeNet->(ImageNet)->AlexNet->VGGNet->GoogLeNet->ResNet->DenseNet
- UvA Tutorial 5: [Inception, ResNet, DenseNet](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html)
- pytorch官方样例仓库 [github](https://github.com/pytorch/examples)
- Mircosoft AI system Lecutures 5~6 分布式训练算法、系统

练习

- cs231n Backprop Review Session [Slides](https://cs231n.stanford.edu/slides/2025/section_2.pdf), [[Colab]](https://colab.research.google.com/drive/1yjxfAugU5JrbgCb1TcCbXDMCKE_G_P-e)
- cs231n PyTorch Review Session on [[Colab]](https://colab.research.google.com/drive/1Dl_Xs5GKjEOAedQUVOjG_fPMB1eXtLSm)
- AI system [Lab2](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/README.md)： 实现了一个C++张量运算，profiling
- AI system [Lab3](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab3/README.md)： 实现了一个cuda算子，profiling
- cs231n Assignment 1 on Colab: [Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets](https://cs231n.github.io/assignments2025/assignment1/)
- Microsoft Lab4 (Optional): 使用 horovod 分布式并行训练
- Microsoft Lab5 (Optional)：使用docker容器部署训练和推理任务

## 2. Language Model

embedding：word2vec
RNN/LSTM: encoder, decoder
Attention, positional encoding, Transformers

Slides:

- CSE538 sp25 (5) Introduction to Language Modeling 3-4
- CSE538 sp25 (6) Neural Networks and RNNs 3-24
- CSE538 sp25 (7) Attention and Transformer LMs 4-2
- cs231n Lecture 7: RNNs and LSTMs
- cs231n Lecture 8: Attention and Transformers
- 线性注意力模型 Songlin Yang's Blog: [DeltaNet Explained Part 1](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- RWKV，Mamba

练习

- cs231n RNNs & Transformers [[Colab]](https://colab.research.google.com/drive/1mC5CWwekbZ2NrYv6Zfpuv55z8DuOZXVP?usp=sharing), [slides](https://cs231n.stanford.edu/slides/2025/section_5.pdf)
- UvA Tutorial [Transformers and MHAttention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

作业

- RNN 二选一

  - cs231n Assignment 2 on Colab: [Batch Normalization, Dropout, Convolutional Nets, Network Visualization, Image Captioning with RNNs](https://cs231n.github.io/assignments2025/assignment2/)
  - CSE538 Assignment 2: RNN LM

- Transformer 二选一
  - cs231n Assignment 3 on Colab or Locally: [Image Captioning with Transformers, Self-Supervised Learning, Diffusion Models, CLIP and DINO Models](https://cs231n.github.io/assignments2025/assignment3/)
  - CSE538 Assignment 3: Transformer LM

阅读

- Seq2seq https://github.com/harvardnlp/seq2seq-attn
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Lilian Weng blog: Attention? Attention](https://lilianweng.github.io/posts/2018-06-24-attention/)
- visionbook.mit.edu: [transformers](https://visionbook.mit.edu/transformers.html)
- ViT: Transformers for Image Recognition [[Blog]](https://research.google/blog/transformers-for-image-recognition-at-scale/?m=1)
- DETR: End-to-End Object Detection with Transformers [[Paper]](https://arxiv.org/abs/2005.12872) [[Blog]](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/) [[Video]](https://www.youtube.com/watch?v=utxbUlo9CyY)
- DINO: Emerging Properties in Self-Supervised Vision Transformers [[Paper]](https://arxiv.org/abs/2104.14294) [[Blog]](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training) [[Video]](https://youtu.be/h3ij3F3cPIk) [[Code]](https://github.com/facebookresearch/detr)
- Github huggingface/transformers [notebooks](https://github.com/huggingface/transformers/tree/main/notebooks), [examples](https://github.com/huggingface/transformers/tree/main/examples), [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md)
- OpenAI CLIP (Contrastive Language-Image Pre-Training), [[Github]](https://github.com/openai/CLIP), [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)
- open_clip: An open source implementation of CLIP. [Github](https://github.com/mlfoundations/open_clip)

## 3. Generative Model 生成模型

[李宏毅 MLDS 2018](https://speech.ee.ntu.edu.tw/~hylee/mlds/2018-spring.php)

- GAN Introduction
- Conditional GAN
- Unsupervised Conditioned GAN
- Theory
- General Framework
- WGAN & EBGAN

KAIST [CS492D](https://mhsung.github.io/kaist-cs492d-fall-2024/)

- lecture 2 GAN-VAE
- lecture 3 DDPM1
- lecture 4 DDPM2
- lecture 7 Classifier-Free Guidance，Latent Diffusion，ControlNet，LoRA
- ODE/DPM solver, score-matching，flow matching

CS231n

- Lecture 13: Generative Models 1 VAE
- Lecture 14: Generative Models 2 GAN, AR, Diffusion
- Lecture 17: Robot Learning
  - Deep Reinforcement Learning
  - Model Learning
  - Robotic Manipulation

练习

- VAE: Auto-encoding Variational Bayes, [pytorch/examples/vae](https://github.com/pytorch/examples/blob/main/vae/README.md)
- GAN: [pytorch/examples/dcgan](https://github.com/pytorch/examples/tree/main/dcgan)
- hugginface/diffusers [[Github]](https://github.com/huggingface/diffusers), [Tutorial: Diffusion models from scratch](https://huggingface.co/learn/diffusion-course/unit1/3)

作业：

- [MLDS 2018](https://speech.ee.ntu.edu.tw/~hylee/mlds/2018-spring.php): Assignment 3.1 3.2
- CS492D: [Assignments 1: ddpm](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment1-DDPM)
- CS492D: [Assignments 2: ddim/CFG](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment2-DDIM-CFG)
- CS492D: [Assignments 3: ControlNet and LoRA](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment3-ControlNet-LoRA)
- CS492D: [2025 Assignment 2 (DPMSolver)](https://github.com/KAIST-Visual-AI-Group/Diffusion-2025-Assignment2-DPMSolver)
- CS492D: [Assignment7 (Flow)](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment7-Flow)

阅读：

- Yunfan's Blog: [ELBO — What & Why](https://yunfanj.com/blog/2021/01/11/ELBO.html)
- Song Yang's blog [score-matching](https://yang-song.net/blog/2021/score/) , [colab tutorial in pytorch](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)
- KAIST CS492d [ Diffusion useful resources](https://mhsung.github.io/kaist-cs492d-fall-2024/#useful-resources)
- Lil's Blog: [Video Generation Modeling from Scratch](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
- MIT diffusion 2025:[ Generative Robotics - Guest lecture by Benjamin Burchfiel (Toyota Research)](https://youtu.be/7tsCN2hRBMg)
- William Peebles and Saining Xie, DiT: Scalable Diffusion Models with Transformers, [[Github]](https://github.com/facebookresearch/DiT)
- Latent-Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models, [[Github]](https://github.com/CompVis/latent-diffusion)
- ICML 2025 Benjamin Burchfiel: [Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets](https://weirdlabuw.github.io/uwm/)
- Toyota Research robotics [blogs](https://medium.com/toyotaresearch/subpage/e87b98e0bd5c), [publications](https://www.tri.global/publications)

## 后续

练习

- UvA DL Tutorial [11: Normalized Flow Image Modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html)
- UvA DL Tutorial [15: Vision Transformers](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html)
- UvA DL Tutorial [Dynamic systems Neural ODEs](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_systems/dynamical_systems_neural_odes.html)
- [MIT Diffusion Labs](https://diffusion.csail.mit.edu/2025/index.html)
  - MIT Diffusion Lab1: working with SDEs
  - MIT Diffusion Lab2: Flow matching and Score matching
  - MIT Diffusion Lab3: conditional image generation

视频生成扩散模型训练与推理

- TurboDiffusion: 100–200× Acceleration for Video Diffusion Models, [[Paper]](https://github.com/thu-ml/TurboDiffusion?tab=readme-ov-file)
- rCM: Score-Regularized Continuous-Time Consistency Model, [[Code]](https://github.com/NVlabs/rcm)
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention, [[Code]](https://github.com/thu-ml/SLA)
- 阿里 Wan2.1: Open and Advanced Large-Scale Video Generative Models, [Wan2.1 Github](https://github.com/Wan-Video/Wan2.1)，[Wan2.2](https://github.com/Wan-Video/Wan2.2)
- Google: Lumiere - A Space-Time Diffusion Model for Video Generation, [Project](https://lumiere-video.github.io/)

## 其他内容

- [Dive into Deep Learning](https://d2l.ai/index.html)
- [Berkeley CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/)
- [Transformers as a Computational Model (UC Berkeley, Simons Institute)](https://www.youtube.com/playlist?list=PLgKuh-lKre11RuxGM038u0OSxVdCicIMF)
- [CS287 Advanced Robotics at UC Berkeley Fall 2019](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNBPJdt8WamRAt4XKc639wF)
- [Github Awesome AI](https://github.com/owainlewis/awesome-artificial-intelligence)
- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)
- [推荐系统相关论文汇总](https://github.com/tangxyw/RecSysPapers)
- [谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】](https://zhuanlan.zhihu.com/p/63186101)
- [看Google如何实现Wide & Deep模型(1)~(4)](https://zhuanlan.zhihu.com/p/47293765)
- [涨点利器：推荐系统中对双塔模型的各种改造升级](https://zhuanlan.zhihu.com/p/468875898)
