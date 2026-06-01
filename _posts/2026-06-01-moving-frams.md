---
layout: post
title: Moving Frames
date: 2026-06-01 00:00:00
description: notes of Riemannian Geometry in an Orthonormal Frame
tags: differential forms
categories: math
---

理解活动标架法才能理解嘉当和陈省身的微分几何思想。

Riemannian Geometry in an Orthonormal Frame, 这本书是嘉当工作的一个不错的介绍，由 Eliè Cartan 1926-1927 年在索邦的讲义整理而来，由 Vladislav V. Goldberg 从俄语版翻译而来，陈省身先生做序。

至于 Moving Frames for Beginners, 据说完全不适合初学者。

# 前言

> 黎曼的几何是对高斯曲面理论的一个推广。他引入了曲率张量，并推导了常曲率的度量的共形形式。后者是他论文中的唯一公式。
>
> 人们认识到基本的问题是形式的问题，即，两个黎曼度量相差一个坐标变换的条件。这个问题在1870年被 Christoffel 和 Lipschitz 用不同方法分别解决了。其中 Christoffel 的方法是后来 Ricci 的张量分析的前提。
>
> 根据 Levi-Civita 的定义，黎曼几何中张量丛的绝对微分的存在性是一个优美且基本的事实。但是，流形上一个更基础的分析工具是 Cartan 1922 年引入的 exterior calculus（外微积分），其作为可微结构的一个结果存在。
>
> 当我们做解析的欧氏几何时，我们更倾向于一个正交坐标系，而不是一个一般的笛卡尔坐标系。Cartan 给黎曼几何实现了这一点。在这个意义下这本书不需要更多的推荐了。
>
> S. S. Chern 陈省身

Chern 的前言总给人高屋建瓴的感觉，对于微分几何的认识有着超出常人的深度。在战火纷飞的年代，他赴欧洲深造，有幸接受嘉当的指导，从而掌握了活动标架法。不知道为什么书中总说当时嘉当的工作人们很难读懂，所以后来是 Chern 将嘉当的思想介绍给美国的数学家，并且在普林斯顿给出高维的高斯-博内-陈定理的六页证明，建立整体微分几何，这些都属于水到渠成。

书的内容不少，总共25个 section。

# 第一部分 Preliminaries

4 个 section 介绍了 Moving Frames, Theory of Pfaffian Forms, Pfaffian Differential Equations 系统的积分，和推广。

活动标架法的想法非常简单直接，先考虑欧氏空间里任一点 $M$ 放置一个右手系直角三棱锥，所有的正交三棱锥由 6 个参数确定：顶点的三个坐标和相对一个固定点 $O$ 处的固定正交三棱锥旋转的欧拉角
$$u_1,u_2,\cdots,u_6.$$

这个三棱锥𝒯可以通过顶点的位置向量 $\mathbf{M}= \overrightarrow{\text{OM}}$ 和其轴的三个互相正交单位向量
$$I_1,I_2,I_3$$
定义。

定义一个无穷小近处 $M'$ 点的三棱锥 $\mathcal{T}'$。新三棱锥 $\mathcal{T}'$ 相对这个三棱锥 $\mathcal{T}$ 的位置通过顶点位置向量的增量确定，
$$\mathbf{M}' -\mathbf{M}= d\mathbf{M},$$

以及轴的单位向量，

$$ \mathbf{M}' -\mathbf{M}= d\mathbf{M},$$

$Μ'$ and $Ι'_k$

$$\mathbf{I}'_k -\mathbf{I}_k = d\mathbf{I}_k, \quad k = 1, 2, 3.$$

无穷小邻域的三棱锥表示为 $M'$ and $I'_k$。

$d M$ and $d I_k$ 投影到 $I_k$ of the trihedron $\mathcal{T}$，然后我们得到 $\mathcal{T}$ 的位移的无穷小表示

$$
\begin{equation}
  \left\{\begin{array}{l}
    d M = \omega^1 I_1 + \omega^2 I_2 + \omega^3 I_3\\
    d I_1 = \omega_1^1 I_1 + \omega_1^2 I_2 + \omega_1^3 I_3\\
    d I_2 = \omega_2^1 I_1 + \omega_2^2 I_2 + \omega_2^3 I_3\\
    d I_3 = \omega_3^1 I_1 + \omega_3^2 I_2 + \omega_3^3 I_3
  \end{array}\right.
\end{equation}
$$

或简写为

$$\text{dM} = \omega^i I_i, \quad d I_i = \omega_i^j I_j, \quad i, j = 1, 2, 3$$

$\omega^i$ projections of the differential $\text{dM}$ on the axes $I_k$

他们是线性微分形式，对 $\omega_i^j$ 来说也对：

$$
\begin{equation}
  \left\{\begin{array}{l}
    \omega^i = \Gamma_{\alpha}^i d u^{\alpha}\\
    \omega_i^j = \Gamma_{i \alpha}^j d u^{\alpha}
  \end{array}\right.
\end{equation}
$$

## Relations among 1-forms of an orthonormal frame.

$\omega_i$ 和 $\omega_i^j$ 不能任意给定。首先，他们由 $I_k$ 的正交性联系起来

$$
\begin{equation}
  \left\{\begin{array}{l}
    I_1^2 = 1, \quad I_2^2 = 1, \quad I_3^2 = 1\\
    I_1 \cdot I_2 = 0, \quad I_2 \cdot I_3 = 0, \quad I_3 \cdot I_1 = 0
  \end{array}\right.
\end{equation}
$$

对两边求微分，我们得到

$$I_i \cdot d I_i = 0, \quad I_j \cdot d I_i + I_i \cdot d I_j = 0, \quad i, j = 1, 2, 3$$

代入 $dI_i$ 且考虑(3)我们发现

$$
\begin{equation}
  w_i^i = 0, \quad \omega_i^j + \omega_j^i = 0, \quad i \neq j
\end{equation}
$$

（这就是反对称性！）

因此只有3个独立的 1-form $w_2^3, w_3^1, w_1^2$. 他们也被叫做三棱锥的旋转分量。$\omega_i$ 被叫做平移分量。

这本书真是基本没有多余的话，逻辑一环套一环。

## 给定一族三棱锥，计算这些分量

给定 $M, I_k$，这些分量很容易计算。略去。

## 移动标架 Moving Frames

除了正交三棱锥，我们可以替换为一个任意斜三棱锥满足
$$(e_1,e_2,e_3)\neq 0$$

$$
\begin{equation}
  \left\{\begin{array}{l}
    d M = \omega^1 e_1 + \omega^2 e_2 + \omega^3 e_3\\
    d e_i = \omega_i^1 I_1 + \omega_i^2 I_2 + \omega_i^3 I_3
  \end{array}\right.
\end{equation}
$$

此时，正交性、反对称性不满足。

## Line element of the space

引入一些记号

$$
\begin{equation}
  e_i \cdot e_j = g_{i j}, \quad g_{i j} = g_{j i}
\end{equation}
$$

$$d s^2 = (d M)^2$$

$$
\begin{align}
    d s^2 & = (e_1)^2 (\omega^1)^2 + \cdots + 2 e_1 \cdot e_2 \omega^1
    \omega^2 + \cdots \nonumber\\
    & = g_{1 1} (\omega^1)^2 + g_{2 2} (\omega^2)^2 + g_{3 3} (\omega^3)^2
    \nonumber\\
    & + 2 g_{1 2} \omega^1 \omega^2 + 2 g_{2 3} \omega^2 \omega^3 + 2 g_{3 1}
    \omega^3 \omega^1 \nonumber
  \end{align}
$$

$$
\begin{equation}
  d s^2 = g_{i j} \omega^i \omega^j
\end{equation}
$$

这个式子 n 维欧氏空间也成立。

内积对两个向量 $X = X^i e_i, \quad Y = Y^i e_i$ 可以类似计算

$$
\begin{equation}
  X \cdot Y = e_i \cdot e_j X^i X^j = g_{i j} X^i X^j
\end{equation}
$$

两向量间夹角 $\varphi$ 的余弦值

$$
\begin{equation}
  \cos \varphi = \frac{g_{i j} X^i Y^j}{\sqrt{g_{i j} X^i X^j} \sqrt{g_{i j}
  Y^i Y^j}}
\end{equation}
$$

## Contravariant and covariant components

$$
\begin{equation}
  X_i = g_{i k} X^k .
\end{equation}
$$

The component $X^k$ are called the contravariant component of a vector $X$,
and the component $X_k$ are called the covariant components of a vector $X$.

# Pfaffian Forms

## Differentiating in a given direction.

Suppose that a Pfaffian form

$$\omega = a_1 d x^1 + a_2 d x^2 + \cdots + a_n d x^n$$

is given. Usually the differentials $d x^1, \ldots, d x^n$ are considered as
new independent variables.

# 个人感悟

大师的出发点和技巧看起来有一些惊人的简洁。黎曼博士论文中推导复变函数的性质，嘉当用直角三棱锥用微元法，Minkowski，Voronoy、Delaunay、Alexandov，Chern, Calabi-Yau 等等。

数学家们或者几何学家们常问的问题就是，什么数据可以决定一个什么的存在，条件是什么？

三条边长满足三角不等式组时确定一个欧氏三角形，三个和为180的角度确定一组相似三角形，给定法向量和面积确定凸多面体的 Minkowski 问题，背后是实的 Monge-Ampère equation。给定拓扑条件和曲率条件，Karler 流形的度量能否确定？

这是一种压缩吗？还是一种关于本质、事物之间深刻联系的观点？

据说嘉当是为了研究这种形式的积分而发现了反对称性的外微分的重要性的。知乎有篇回答，说嘉当发现反对称重要，但是不知道为什么重要，感觉应该是不对，别的回答解释过为什么嘉当发展了外微分的理论，比如要研究 Pfaffian 形式的微分方程。后面就先略去。
