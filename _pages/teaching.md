---
layout: page
permalink: /teaching/
title: Learn From Famous AI Courses
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
- [Dive into Deep Learning](https://d2l.ai/index.html)  
- [Berkeley CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/)  

## 1. AI system与图像分类模型
运行环境：本地nvidia gpu，或colab T4 gpu

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
    * 安装gpu driver: nvidia-smi
    * 安装cuda toolkit: nvcc compiler
    * 安装miniconda，创建env，安装pytorch，验证gpu available
	* linux系统: 查看 nvidia-fabric-manager状态
	* jupyter notebook
	* 安装docker

练习  
cs231n python/numpy练习 Colab

第二讲：  
CSE 538 maximum entropy classifier (Logistic regression)  
cs231n Lecture 4: Neural Networks and Backpropagation

练习  
pytorch.org tutorial with examples [tensor, function, auto grad, module](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html) 
cs231n Backprop Review Session Colab

第三讲：  
cs231n Lecture 5: Image Classification with CNNs
[shervine blog: 图片识别模型的演化](https://stanford.edu/~shervine/blog/evolution-image-classification-explained)：  
	LeNet->(ImageNet)->AlexNet->VGGNet->GoogLeNet->ResNet->DenseNet

练习  
- cs231n PyTorch Review Session on Colab  
- [UvA DL Notebooks Tutorial 2: Introduction to PyTorch](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html)
- [pytorch nn tutorial](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html) mnist 图片识别  
- [AI system Lab1](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab1/README.md)： mnist 样例，模型可视化，逐layer性能profiling
- [AI system Lab2](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/README.md)： 实现一个张量运算C++，profiling
- [AI system Lab3](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab3/README.md)： cuda实现和优化，profiling
- Lab4 Optional: 使用 horovod 分布式并行训练
- Lab5 Optional：使用docker容器部署训练和推理任务
- cs231n Assignment 1 on Colab: [Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets](https://cs231n.github.io/assignments2025/assignment1/)

阅读：  
- [tensorflow从0到N：反向传播的推导](https://github.com/EthanYuan/TensorFlow-Zero-to-N/blob/master/TensorFlow%E4%BB%8E0%E5%88%B0N/TensorFlow%E4%BB%8E0%E5%88%B01/10-NN%E5%9F%BA%E6%9C%AC%E5%8A%9F%EF%BC%9A%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%9A%84%E6%8E%A8%E5%AF%BC.md) 
- [UvA Tutorial 5: Inception, ResNet, DenseNet](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html)
pytorch样例 https://github.com/pytorch/examples
- AI system 5~6 分布式训练算法、系统

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

练习  
- cs231n RNNs & Transformers Colab
- [UvA Tutorial Transformers and MHAttention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
- cs231n Assignment 2 on Colab: [Batch Normalization, Dropout, Convolutional Nets, Network Visualization, Image Captioning with RNNs](https://cs231n.github.io/assignments2025/assignment2/)
- CSE538 Assignment 2: RNN LM
- CSE538 Assignment 3: Transformer LM

阅读  
- Seq2seq https://github.com/harvardnlp/seq2seq-attn
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)  
- [Lilian Weng blog: Attention? Attention](https://lilianweng.github.io/posts/2018-06-24-attention/)
- ViT: Transformers for Image Recognition [Blog](https://research.google/blog/transformers-for-image-recognition-at-scale/?m=1)
- Github Huggingface/transformers [notebooks](https://github.com/huggingface/transformers/tree/main/notebooks), [examples](https://github.com/huggingface/transformers/tree/main/examples)

## 3. Generative Model 生成模型
李宏毅 GAN Introduction, Conditional GAN, Unsupervised Conditioned GAN, Theory, General Framework, WGAN & EBGAN

KAIST CS492D
lecture 2 GAN-VAE
lecture 3 DDPM1
lecture 4 DDPM2
lecture 7 Classifier-Free Guidance，Latent Diffusion，ControlNet，LoRA
ODE/DPM solver, score-matching，flow matching

练习  
- VAE: Auto-encoding Variational Bayes https://github.com/pytorch/examples/blob/main/vae/README.md
- GAN: [李宏毅MLDS 2018](https://speech.ee.ntu.edu.tw/~hylee/mlds/2018-spring.php) Assignment 3.1 3.2
- [pytorch/examples/dcgan](https://github.com/pytorch/examples/tree/main/dcgan)
- cs231n Assignment 3 on Colab or Locally: [Image Captioning with Transformers, Self-Supervised Learning, Diffusion Models, CLIP and DINO Models](https://cs231n.github.io/assignments2025/assignment3/)

CS492D:  
- [Assignments 1: ddpm](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment1-DDPM)
- [Assignments 2: ddim/CFG](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment2-DDIM-CFG)
- [Assignments 3: ControlNet and LoRA](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment3-ControlNet-LoRA)
- [2025 Assignment 2 (DPMSolver)](https://github.com/KAIST-Visual-AI-Group/Diffusion-2025-Assignment2-DPMSolver)
- [Assignment7 (Flow)](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment7-Flow)

阅读  
- [Yunfan's Blog: ELBO — What & Why](https://yunfanj.com/blog/2021/01/11/ELBO.html)
- [Song Yang's blog score-matching](https://yang-song.net/blog/2021/score/) , [colab tutorial in pytorch](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)
- [KAIST CS492d - Diffusion useful resources](https://mhsung.github.io/kaist-cs492d-fall-2024/#useful-resources) 
- [MIT diffusion 2025: Generative Robotics - Guest lecture by Benjamin Burchfiel (Toyota Research)](https://youtu.be/7tsCN2hRBMg) 
- [ICML 2025 Benjamin Burchfiel: Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets](https://weirdlabuw.github.io/uwm/) 
- Toyota Research robotics [blogs](https://medium.com/toyotaresearch/subpage/e87b98e0bd5c), [publications](https://www.tri.global/publications)

## 后续
练习  
- [UvA DL Tutorial 11: Normalized Flow Image Modeling](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html)
- [UvA DL Tutorial 15: Vision Transformers](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html)
- [Dynamic systems Neural ODEs](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_systems/dynamical_systems_neural_odes.html)
- [MIT Diffusion Labs](https://diffusion.csail.mit.edu/2025/index.html)
    * MIT Diffusion Lab1: working with SDEs
    * MIT Diffusion Lab2: Flow matching and Score matching
    * MIT Diffusion Lab3: conditional image generation

## 其他内容

- [Transformers as a Computational Model (UC Berkeley, Simons Institute)](https://www.youtube.com/playlist?list=PLgKuh-lKre11RuxGM038u0OSxVdCicIMF)
- [CS287 Advanced Robotics at UC Berkeley Fall 2019](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNBPJdt8WamRAt4XKc639wF)

- [Github Awesome AI](https://github.com/owainlewis/awesome-artificial-intelligence)

- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)

- [推荐系统相关论文汇总](https://github.com/tangxyw/RecSysPapers) 
- [谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】](https://zhuanlan.zhihu.com/p/63186101) 
- [看Google如何实现Wide & Deep模型(1)~(4)](https://zhuanlan.zhihu.com/p/47293765) 
- [涨点利器：推荐系统中对双塔模型的各种改造升级](https://zhuanlan.zhihu.com/p/468875898)
