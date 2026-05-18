---
layout: post
title: Introduction of Alexandrov's Convex Polyhedra
date: 2026-05-18 17:00:00
description: note of this book
tags: convex
categories: math
---

Alexandrov 把这本书献给了 Boris Nikolaevich Delaunay, my teacher。Delaunay三角剖分很多人都熟悉。Delaunay的导师之一则是 Georgy Voronoy，定义了 Voronoi diagram。

Introduction 中首先介绍了这本书的内容和目的。主要的问题是：哪些数据可以决定一个凸多面体以及多大程度上？这个问题对于多面体上相关的数据如边的长度、面的面积来说又分为两个问题。  
第一个问题是这个数据是否可以确定这个多面体，唯一指在运动不变意义下或其他平凡变换（反射、平移或相似）意义下，就像三条边的长度决定一个运动下不变的三角形，三个角度决定相似不变的三角形。  
第二个问题是数据为了满足存在这样一个凸多面体的必要和充分条件是什么。类比存在一个以 $a$, $b$ 和 $c$ 为边长的三角形的必要和充分条件是三个不等式：$a+b>c$, $b+c>a$, $c+a>b$。因为必要性在所有这些例子里都简单，问题永远都落在充分性，即证明给定数据后满足这些条件的凸多面体的存在性。

之前给出过相关结果的人包括 Cauchy，Minkowski，Olovyanishnikov。
后来丘成桐和郑绍远在证明 Monge-Ampère 方程正则性、解决高维 Minkowski 问题的时候用了 Alexandrov weak solution 的概念，为证明 complex Monge-Ampère 方程解的存在性和卡拉比猜想的证明奠定了基础。

我们一般人很难想象如此简单的数学问题有如此深刻的内涵，一种最非线性的PDE的解法依赖如此基本的几何上的想法。

另外一个扩展的方向则是从Euclidean空间到Sphere 和 Hyperbolic space上。三角形的边长换成测地线长度，凸多面体的面的面积换为极小曲面的面积等等。欧氏空间的 Minkowski 问题对应欧氏最优传输，Sphere and hyperbolic space 上的问题则对应 spherical and hyperbolic optimal transport。
