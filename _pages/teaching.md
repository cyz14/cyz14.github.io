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
第一遍分享，从2026年1月5号开始，至1月26号正式完结。

参考课程包括：

- Microsoft AI System course [[Github]](https://microsoft.github.io/AI-System/)
- Stanford CS231n [Spring 2025](https://cs231n.stanford.edu/schedule.html)
- Stony Brook CSE 538 NLP 2025 Spring [slides](https://www3.cs.stonybrook.edu/~has/CSE538/Slides/)
- 李宏毅 Machine Learning and Having It Deep and Structured, [2018 Spring](https://speech.ee.ntu.edu.tw/~hylee/mlds/2018-spring.php)
- KAIST 492(D) Diffusion Models and Their Applications, [24 Fall](https://mhsung.github.io/kaist-cs492d-fall-2024/)
- MIT EECS 6.S978 Deep Generative Models [Fall 2024](https://mit-6s978.github.io/)
- MIT Diffusion course [2025](https://diffusion.csail.mit.edu/2025/index.html)
- MIT EECS 6.7960 Deep Learning [Fall 2025](https://deeplearning6-7960.github.io/)
- CMU 10-799 Diffusion & Flow Matching, [Spring 2026](https://kellyyutonghe.github.io/10799S26/)
- COMP760 2022: Geometry and Generative Models, [Website](https://joeybose.github.io/Blog/GenCourse)

其他可参考的资源包括

- pytorch官方样例仓库 [github](https://github.com/pytorch/examples)
- huggingface/transformers: [[notebooks]](https://github.com/huggingface/transformers/tree/main/notebooks), [[examples]](https://github.com/huggingface/transformers/tree/main/examples), [[awesome-transformers]](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md)
- hugginface/diffusers [[Github]](https://github.com/huggingface/diffusers), [Tutorial: Diffusion models from scratch](https://huggingface.co/learn/diffusion-course/unit1/3)
- KAIST CS492d [Diffusion useful resources](https://mhsung.github.io/kaist-cs492d-fall-2024/#useful-resources)
- labml.ai: Annotated Deep Learning Paper Implementation, [[Github]](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
- Implement state-of-the-art models from scratch [[PaperCode]](https://papercode.vercel.app/papers)
- Comprehensive toy implementations of the 30 foundational papers recommended by Ilya Sutskever, [[Github]](https://github.com/pageman/sutskever-30-implementations)
- Dive into Deep Learning, [d2l.ai](https://d2l.ai/index.html)

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
  - 安装pytorch，验证 torch.cuda.is_available()
  - 安装docker

练习

- [知乎专栏：安装GPU驱动，CUDA和cuDNN](https://zhuanlan.zhihu.com/p/143429249)
- cs231n python/numpy Review [[Colab]](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb), [Tutorial](https://cs231n.github.io/python-numpy-tutorial/)

第二讲：

- cs231n Lecture 1: Deep Learning for Computer Vision
- CSE 538 Topic 2: maximum entropy classifier (Logistic regression), [[slides]](https://www3.cs.stonybrook.edu/~has/CSE538/Slides/)
- pytorch.org tutorial with examples [tensor, function, auto grad, module](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

练习

- MIT EECS 6.7960 Deep Learning Fall 2025, pytorch tutorial [[colab]](https://colab.research.google.com/drive/1nZg9_wYpVYWS9xZAiSft5_gyluuQpBWY?usp=sharing)
- CSE538 Spring 2025 Assignment 1: Multi-Class Logistic Regression [[Google Drive Download]](https://drive.google.com/drive/folders/1Uj1li3QE-EEl3SVNipbaMqsc6mGWp9_F?usp=drive_link), 参考 [Demo of Logistic Regression with Gradient Descent](https://adithya8.github.io/assets/cse538-sp25/intro_numpy_LogisticRegression.txt)

第三讲：

- cs231n Lecture 2: Image Classification with Linear Classifiers
- cs231n Lecture 3: Regularization and Optimization

练习

- pytorch nn tutorial [What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html): mnist 图片识别
- AI system [Lab1](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab1/README.md)： mnist，模型可视化，逐layer性能profiling

第四讲：

- cs231n Lecture 4: Neural Networks and Backpropagation
- cs231n Lecture 5: Image Classification with CNNs
- cs231n Lecture 6: Training CNNs and CNN Architectures

阅读：

- Kaiming He CVPR25 talk: [Workshop: What's After Diffusion?](https://people.csail.mit.edu/kaiming/cvpr25talk/cvpr2025_meanflow_kaiming.pdf)
- cs231n [Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- AlexNet [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- GoogLeNet [arxiv](https://arxiv.org/abs/1409.4842)
- ResNet [arxiv](https://arxiv.org/abs/1512.03385)
- shervine blog: [Cheatsheet CNN](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- Mircosoft AI system Lecutures 5~6 分布式训练算法、系统

练习

- cs231n Backprop Review Session [Slides](https://cs231n.stanford.edu/slides/2025/section_2.pdf), [[Colab]](https://colab.research.google.com/drive/1yjxfAugU5JrbgCb1TcCbXDMCKE_G_P-e)
- cs231n PyTorch Review Session on [[Colab]](https://colab.research.google.com/drive/1Dl_Xs5GKjEOAedQUVOjG_fPMB1eXtLSm)
- AI system [Lab2](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/README.md)： 实现了一个C++张量运算，profiling
- AI system [Lab3](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab3/README.md)： 实现了一个cuda算子，profiling
- pytorch 使用 tensorboard 查看降维算法 embedding [tensorboard tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- Microsoft Lab4 (Optional): 使用 horovod 分布式并行训练
- Microsoft Lab5 (Optional)：使用 docker 容器部署训练和推理任务

作业

- cs231n Assignment 1 on Colab: [Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets](https://cs231n.github.io/assignments2025/assignment1/)

## 2. Language Model

参考教材: Speech and Language Processing (3rd ed. draft), Dan Jurafsky and James H. Martin, [[Book]](https://web.stanford.edu/~jurafsky/slp3/)

Slides:

- CSE538 sp25 (5) Introduction to Language Modeling 3-4
- CSE538 sp25 (6) Neural Networks and RNNs 3-24
- CSE538 sp25 (7) Attention and Transformer LMs 4-2
- cs231n Lecture 7: RNNs and LSTMs
- cs231n Lecture 8: Attention and Transformers
- cs231n Lecture 9: Detection, Segmentation, Visualization, and Understanding
- cs231n Lecture 10: Video Understanding
- Related topics about LLM: CLIP, Chain-of-Thought, Reinforcement Learning, Test Time Learning, Linear Attention, State Space Model, Large Language Diffusion Models

练习

- Karpathy's char-rnn of Shakespear [[Github]](https://github.com/karpathy/char-rnn)
- cs231n RNNs & Transformers [[Colab]](https://colab.research.google.com/drive/1mC5CWwekbZ2NrYv6Zfpuv55z8DuOZXVP?usp=sharing), [slides](https://cs231n.stanford.edu/slides/2025/section_5.pdf)
- Karpathy's build-nanogpt [[Github]](https://github.com/karpathy/build-nanogpt)

作业

- RNN

  - CSE538 Assignment 2: RNN LM. Instructions [Google Drive](https://drive.google.com/drive/folders/1snFSimiWyiKAxPYXMfB-mykTIVaYJPPu?usp=sharing)
  - cs231n Assignment 2 on Colab: [Batch Normalization, Dropout, Convolutional Nets, Network Visualization, Image Captioning with RNNs](https://cs231n.github.io/assignments2025/assignment2/)

- Transformer

  - CSE538 Assignment 3: Transformer LM. Instructions [Google Drive](https://drive.google.com/drive/folders/1jhslUZkQ9hFWPoGrv6OqDwih0DYLMluV?usp=sharing)
  - cs231n Assignment 3 on Colab or Locally: [Image Captioning with Transformers, Self-Supervised Learning, Diffusion Models, CLIP and DINO Models](https://cs231n.github.io/assignments2025/assignment3/)

- 从零实现语言模型 Stanford CS336: Language Modeling from Scratch, Spring 2025, [5 Assignments](https://stanford-cs336.github.io/spring2025/)
  - Implement tokenizer, model architecture, optimizer
  - Profile and benchmark the model and layers, optimize Attention with your own Triton implementation of FlashAttention2.
  - Build a memory-efficient, distributed version of the Assignment 1 model training
  - Scaling
  - Data: clean, filtering and deduplication
  - supervised finetuning and reinforcement learning

阅读

- harvardnlp [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- Jay Alammar: [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Lilian's blog: [Attention? Attention](https://lilianweng.github.io/posts/2018-06-24-attention/)
- ViT: Transformers for Image Recognition [[Blog]](https://research.google/blog/transformers-for-image-recognition-at-scale/?m=1), Image captioning with ViT [[Github]](https://github.com/inuwamobarak/Image-captioning-ViT)
- DINO: Emerging Properties in Self-Supervised Vision Transformers [[Paper]](https://arxiv.org/abs/2104.14294) [[Blog]](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training) [[Video]](https://youtu.be/h3ij3F3cPIk) [[Code]](https://github.com/facebookresearch/detr)
- Vision Transformer: When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations, [Code](https://github.com/google-research/vision_transformer)
- RWKV, by Peng Bo, [Project Website](https://www.rwkv.com/), [知乎: 推荐几个RWKV的Chat模型...](https://zhuanlan.zhihu.com/p/618011122)
- Mamba, by Albert Gu, Tri Dao, [arxiv](https://arxiv.org/abs/2312.00752), Albert Gu [video: Efficiently Modeling Long Sequences with Structured State Spaces](https://www.youtube.com/watch?v=luCBXCErkCs)
- CSDN/v_July_v: [一文通透想颠覆Transformer的Mamba：从SSM、HiPPO、S4到Mamba](https://blog.csdn.net/v_JULY_v/article/details/134923301)
- DeltaNet, Songlin Yang's Blog: [DeltaNet Explained Part 1](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- Maarten Grootendorst: [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
- OpenAI CLIP (Contrastive Language-Image Pre-Training), [[Github]](https://github.com/openai/CLIP), [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb). open_clip: An open source implementation of CLIP. [Github](https://github.com/mlfoundations/open_clip)
- CodeFusion: A Pre-trained Diffusion Model for Code Generation, [arxiv](https://arxiv.org/abs/2310.17680)
- Large Language Diffusion Models, [arxiv](https://arxiv.org/abs/2502.09992), [LLaDA-demo](https://ml-gsai.github.io/LLaDA-demo/), [[LLaDA 2.0]](https://github.com/inclusionAI/LLaDA2.0)
- Mercury: Ultra-Fast Language Models Based on Diffusion, [Inception Labs](https://www.inceptionlabs.ai/), [arxiv](https://arxiv.org/abs/2506.17298)

## 3. Generative Model 生成模型

参考教材：Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji, Stefano Ermon. The Principles of Diffusion Models - From Origins to Advances [[Book Page]](https://the-principles-of-diffusion-models.github.io/)

KAIST CS492D [Fall 24](https://mhsung.github.io/kaist-cs492d-fall-2024/)

- lecture 2 GAN-VAE
- lecture 3 DDPM1
- lecture 4 DDPM2
- lecture 5 DDIM & score matching
- lecture 6 DDIM & CFG
- lecture 7 CFG，Latent Diffusion，ControlNet，LoRA
- lecture 14: DPM/ODE solver
- lecture 15 & 16: Flow matching

Chapter 6 of [The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/) gives a unified and systematic lens on diffusion models. The training objective of diffusion models commonly share the following template form:

$\mathcal{L} (\phi) := \mathbb{E}_{x_0, \epsilon} 
\underbrace{\mathbb{E}_{p_{\text{time}} (t)}}_{{\begin{array}{l}
  \text{time}\\
  \text{distribution}
\end{array} }}  \left[ \underbrace{\omega (t)}_{{\begin{array}{l}
  \text{time}\\
  \text{weighting}
\end{array}}}  \underbrace{\| \text{NN}_{\phi} (x_t, t) - (A_t x_0 + B_t
\epsilon) \|^2_2}_{\text{MSE} \quad \text{part}}  \right]$

All prediction types share a common regression target of the form

$$ \text{Regression Target}= A_t x_0 + B_t \epsilon $$

The coefficients $A_t $ and $B_t$ depend on both the chosen prediction type and the schedule $(\alpha_t, \sigma_t)$. The relationships are summarize in the book's Table 6.1.

| Regression Target=   | $A_t x_0 + B_t \epsilon$ |                        |
| -------------------- | ------------------------ | ---------------------- |
|                      | $A_t$                    | $B_t$                  |
| Clean                | 1                        | 0                      |
| Noise                | 0                        | 1                      |
| Conditional Score    | 0                        | $- \frac{1}{\sigma_t}$ |
| Conditional Velocity | $\alpha_t'$              | $\sigma_t'$            |

阅读：

- MLDS [Spring 2018](https://speech.ee.ntu.edu.tw/~hylee/mlds/2018-spring.php): GAN Introduction, Conditional GAN, Unsupervised Conditioned GAN, Theory, General Framework, WGAN & EBGAN
- Diederik P. Kingma, Max Welling, An Introduction to Variational Autoencoders, 2019, [[arvix]](https://arxiv.org/abs/1906.02691)
- 苏剑林，变分自编码器（一）：原来是这么一回事, [kexue.fm](https://kexue.fm/archives/5253), 变分自编码器（二）：从贝叶斯观点出发, [kexue.fm](https://kexue.fm/archives/5343)
- Yunfan's Blog: [ELBO — What & Why](https://yunfanj.com/blog/2021/01/11/ELBO.html)
- 2021 Song Yang's blog [score-matching](https://yang-song.net/blog/2021/score/) , [colab tutorial in pytorch](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)
- 2021 Lilian's Blog, [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- 2023 The Fokker-Planck Equation and Diffusion Models, [Blog](https://www.peterholderrieth.com/blog/2023/The-Fokker-Planck-Equation-and-Diffusion-Models/)
- 2024 An Introduction to Flow Matching, [Blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- 2015 Sohl-Dickstein et al. Deep Unsupervised Learning using Nonequilibrium Thermodynamics [[arxiv]](https://arxiv.org/pdf/1503.03585), [[Code]](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)
- 2019 Yang Song, Stefano Ermon, Generative Modeling by Estimating Gradients of the Data Distribution, [[arxiv]](https://arxiv.org/abs/1907.05600)
- 2020 Jonathan Ho, Ajay Jain, Pieter Abbeel, Denoising Diffusion Probabilistic Models, [[arxiv]](https://arxiv.org/abs/2006.11239), [[Code]](https://github.com/hojonathanho/diffusion)
- 2021, Latent-Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models, [[Github]](https://github.com/CompVis/latent-diffusion)
- 2022 William Peebles and Saining Xie, DiT: Scalable Diffusion Models with Transformers, [[arxiv]](https://arxiv.org/abs/2212.09748), [[Github]](https://github.com/facebookresearch/DiT)

练习

- VAE: Auto-encoding Variational Bayes, [pytorch/examples/vae](https://github.com/pytorch/examples/blob/main/vae/README.md), Kingma's mnist [demo](https://dpkingma.com/sgvb_mnist_demo/demo.html)
- GAN: [pytorch/examples/dcgan](https://github.com/pytorch/examples/tree/main/dcgan)
- DDPM: The Annotated Diffusion Model, [huggingface blog](https://huggingface.co/blog/annotated-diffusion)
- labml.ai/Diffusion Models: [[DDPM]](https://nn.labml.ai/diffusion/ddpm/index.html), [[DDIM]](https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html), [[Latent Diffusion Models]](https://nn.labml.ai/diffusion/stable_diffusion/latent_diffusion.html), [[Stable Diffusion]](https://nn.labml.ai/diffusion/stable_diffusion/index.html)
- Facebook, Flow Matching Guide and Code, [[arxiv]](https://arxiv.org/pdf/2412.06264), [[Code]](https://github.com/facebookresearch/flow_matching)

作业：

- CS492D: [Assignments 1: ddpm](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment1-DDPM)
- CS492D: [Assignments 2: ddim/CFG](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment2-DDIM-CFG)
- CS492D: [Assignments 3: ControlNet and LoRA](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment3-ControlNet-LoRA)
- CS492D: [2025 Assignment 2 (DPMSolver)](https://github.com/KAIST-Visual-AI-Group/Diffusion-2025-Assignment2-DPMSolver)
- CS492D: [Assignment7 (Flow)](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment7-Flow)
- MIT Diffusion [Course](https://diffusion.csail.mit.edu/2025/index.html)
  - MIT Diffusion Lab1: working with SDEs
  - MIT Diffusion Lab2: Flow matching and Score matching
  - MIT Diffusion Lab3: conditional image generation

## 后续

### Mesh Generation

- MIT 6s798, [Reading: Application - 3D and Geometry](https://mit-6s978.github.io/schedule.html)
- [Github/topics/3d-mesh-generation](https://github.com/topics/3d-mesh-generation)
- [Github/topics/image-to-3d](https://github.com/topics/image-to-3d)
- TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation [[arxiv]](https://arxiv.org/html/2412.01506) TRELLIS.2, [[arxiv]](https://arxiv.org/html/2512.14692v1)
- Tencent-Hunyuan-3D, [Page](https://hunyuan.tencent.com/3d)
- MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers, CVPR 24, [[Project]](https://nihalsid.github.io/mesh-gpt/)
- DreamFusion: Text-to-3D using 2D Diffusion, [[arxiv]](https://arxiv.org/abs/2209.14988)
- Flexible Isosurface Extraction for Gradient-Based
  Mesh Optimization, [[Project]](https://research.nvidia.com/labs/toronto-ai/flexicubes/)

### Robotics

- CS231n Lecture 17: Robot Learning: Deep Reinforcement Learning, Model Learning, Robotic Manipulation
- MIT 6s798 Deep Generative Models, [Reading: Application - Robotics](https://mit-6s978.github.io/schedule.html)
- MIT diffusion 2025: [Generative Robotics - Guest lecture by Benjamin Burchfiel (Toyota Research)](https://youtu.be/7tsCN2hRBMg)
- 南科大，高等机器人控制 [Bilibili](https://space.bilibili.com/474380277)
- UC Berkeley, [CS287 Advanced Robotics, Fall 2019](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNBPJdt8WamRAt4XKc639wF)
- Benjamin Burchfiel, ICML 2025, [Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets](https://weirdlabuw.github.io/uwm/)
- Tencent-Hunyuan/HY-Motion-1.0: Scaling Flow Matching Models for 3D Motion Generation, [[Code]](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)
- Galaxea/G0: [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA)
- huggingface/LeRobot: Making AI for Robotics more accessible with end-to-end learning, [Github](https://github.com/huggingface/lerobot)
- Physical-Intelligence/openpi Pi0: [Github](https://github.com/Physical-Intelligence/openpi)

### Video Generation

- Lil's Blog: [Video Generation Modeling from Scratch](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
- open sora, [[Github]](https://github.com/hpcaitech/Open-Sora)
- 阿里 Wan2.1: Open and Advanced Large-Scale Video Generative Models, Github [Wan2.1](https://github.com/Wan-Video/Wan2.1)，[Wan2.2](https://github.com/Wan-Video/Wan2.2), 基于 DiffSynth-Studio 的训练[[Code]](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/zh/Model_Details/Wan.md)
- Tencent: Hunyuan [[Project]](https://hunyuan.tencent.com/video/zh?tabIndex=0), [[Github]](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [[Training Section]](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5?tab=readme-ov-file#-training)
- Meta, Movie Gen: A Cast of Media Foundation Models, [[Paper]](https://ai.meta.com/static-resource/movie-gen-research-paper)
- Google: Lumiere - A Space-Time Diffusion Model for Video Generation, [Project](https://lumiere-video.github.io/), Unofficial implementation [[lucidrains/lumiere-pytorch]](https://github.com/lucidrains/lumiere-pytorch), [[kyegomez/LUMIERE]](https://github.com/kyegomez/LUMIERE)
- TurboDiffusion: 100–200× Acceleration for Video Diffusion Models, [[Paper]](https://github.com/thu-ml/TurboDiffusion)
- rCM: Score-Regularized Continuous-Time Consistency Model, [[Code]](https://github.com/NVlabs/rcm)
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention, [[Code]](https://github.com/thu-ml/SLA)
- 港科大&快手可灵 UnityVideo, [[arxiv]](https://arviv.org/abs/2512.07831), [[Project]](https://jackailab.github.io/Projects/UnityVideo), Code 后续开源

### Reinforcement Learning

- Berkeley, [CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/)
- OpenAI, Spinning Up in Deep RL! [Docs](https://spinningup.openai.com/en/latest/)
- 王树森，深度强化学习 [DRL](https://github.com/wangshusen/DRL)

## 其他内容

- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)
- [推荐系统相关论文汇总](https://github.com/tangxyw/RecSysPapers)
- [谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】](https://zhuanlan.zhihu.com/p/63186101)
- [看Google如何实现Wide & Deep模型(1)~(4)](https://zhuanlan.zhihu.com/p/47293765)
- [涨点利器：推荐系统中对双塔模型的各种改造升级](https://zhuanlan.zhihu.com/p/468875898)
- [快手 OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)
