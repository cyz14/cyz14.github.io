---
layout: post
title: Differential Forms, Symmetrizing and Alternating Operators
date: 2026-05-30 01:55:00
description:
tags: differential forms
categories: math
tikzjax: true
---

这两天读 Loring W. Tu 老师的 The Introduction to Manifolds.

第3章 The Exterior Algebra of Multicovectors，第4章 Differential Forms on $\mathbb{R}^n$。直接从这里开始，记录一些重要概念。

## 一些历史

外代数最早是 Hermann Grassmann，一个中学老师发明的。但是他活着的时候没有受到足够的重视，因为当时领头的数学家们比如莫比乌斯和库默尔等人没有理解他的工作。直到20世纪，嘉当意识到微分形式的代数基础就是外代数。从这个意义上讲，我们也是继承了 Grassmann 的学术。

## 对偶空间 Dual Space

$Hom(V,W)$ 表示所有线性映射 $f:V\rightarrow W$的向量空间。

<div class="definition" title="Dual Space">
$V$ 的对偶空间 $V^{\vee}$ 定义为 $V$ 上的实值线性函数:

$$V^{\vee}=Hom(V,\mathbb{R})$$

假设 $V$ 为有限维向量空间。令 $e_1,..,e_n$ 为 $V$ 的一组基底。
每个 $v \in V$ 可以表示为 $v=\sum v^ie_i, v_i\in \mathbb{R}$.

令 $\alpha^i:V\rightarrow \mathbb{R}$ 表示取第 $i$ 个坐标的线性函数， $\alpha^i(v)=v^i$，则
$$\alpha^i(e_j)=\delta_j^i.$$

$\alpha^1,..,\alpha^n$ 构成了 $V^{\vee}$ 的一组基底。对偶空间和 $V$ 有相同的维度。

</div>

## Symmetrizing and Alternating Operators

给一个任意 $k$-linear function $f$ on a vector space $V$，there is a way to make a symmetric $k$-linear function $Sf$ from it
$$(Sf)(v_1,..,v_k)=\sum_{\sigma\in S_k}f(v_{\sigma(1))},\cdots,v_{\sigma(k)})$$
简写为
$$Sf=\sum_{\sigma\in S_k}\sigma f.$$

类似地，有一个办法得到一个反对称的 $k$-linear 函数。定义
$$Af=\sum_{\sigma\in S_k}(\text{sgn } \sigma)\sigma f.$$

<div class="proposition">
$Sf$ 是对称的，$Af$ 是交错（反对称）的。
</div>

<div class="proof">
这是直接验证定义的结果。对任意 $\tau \in S_k$，

$$\tau(Sf) = \sum_{\sigma \in S_k} \tau(\sigma f) = \sum_{\sigma \in S_k} (\tau\sigma) f = Sf,$$

而 $\tau(Af) = \sum (\text{sgn } \sigma)(\tau\sigma) f = (\text{sgn } \tau) Af$。

</div>

## Tensor Product

$f$ be a $k$-linear function and $g$ an $l$-linear function on $V$. Their tensor product is the $(k+l)$-linear function $f \otimes g$ defined by
$$(f \otimes g)(v_1,\cdots,v_{k+l})=f(v_1,\cdots,v_k)g(v_{k+1},\cdots,v_{k+l}).$$

<div class="example" title="Bilinear maps">
$\langle, \rangle : V \times V \rightarrow \mathbb{R}$ a bilinear map on $V$.
Set $g_{i j} = \langle e_i, e_j \rangle \in \mathbb{R}$. By bilinearity, we can
express $\langle, \rangle$ in terms of the tensor product

$$
\langle v, w \rangle = \sum v^i w^j \langle e_i, e_j \rangle = \sum
  \alpha^i (v) \alpha^j (w) g_{i j} = \sum g_{i j} (\alpha^i \otimes
  \alpha^j) (v, w) .
$$

</div>

## The Wedge Product

$f\in A_k(V)$ and $g \in A_l(V)$,

$$f\wedge g=\frac{1}{k!l!}A(f \otimes g).$$

$$f\wedge g=(-1)^{kl}g \wedge f.$$

$$f\wedge g=\frac{1}{k!l!}A(f \otimes g).$$

<div class="proposition" title="Associativity of the wedge product">

$$(f\wedge g)\wedge h=f\wedge (g \wedge h).$$

</div>

<div class="corollary">
$$f\wedge g\wedge h=\frac{1}{k!l!m!}A(f\otimes g \otimes h)$$
</div>

<div class="proposition" title="Wedge product of 1-covectors">
If $\alpha^1,\cdots,\alpha^k$ are linear functions on a vector space $V$ and $v_1,\cdots, v_k \in V$, then
$$(\alpha^1 \wedge \cdots \wedge \alpha^k)(v_1, \cdots, v_k)= \det [\alpha^i(v_j)].$$
</div>

## Differentail 1-Forms and Differential as a Function

切向量基底 $\frac{\partial }{\partial x^i}$,  
对偶空间基底 $dx^i$.

$\langle X_p, f \rangle$

$df = \sum \frac{\partial f}{\partial x^i} dx^i$

a differentail form $\omega$ of degree $k$ or a $k$-form on an open subset $U$ of $\mathbb{R}^n$

an alternating $k$-linear function on the tanget space $T_p(\mathbb{R}^n)$

$\omega_p \in A_k(T_p\mathbb{R}^n)$

$$dx_p^I=dx_p^{i_1}\wedge ... \wedge dx_p^{i_k}, \quad 1\leq i_1 \lt ..\lt i_k \leq n$$

at each point $p$ in $U$, $\omega_p$ is a linear combination
$$\omega_p=\sum a_I(p)dx_p^I,\quad 1\leq i_1 \lt ..\lt i_k \leq n$$

and a $k$-form $\omega$ on $U$ is a linear combination
$$\omega=\sum a_I dx^I$$

Exterior derivative

$f \in C^{\infty}(U)$, $df\in \Omega^1(U)$

$$df=\sum \frac{\partial f}{\partial x^i}d x^i$$

$$d \omega=\sum_I da_I \wedge dx^I=\sum_I (\sum_j \frac{\partial a^I}{\partial x^j}dx^j)\wedge dx^I\in\Omega^{k+1}(U)$$

$A$ be a graded algebra over field $K$.
Antiderivation of the graded algebra $A$ is a $K$-linear map $D:A->A$ such that for $a\in A^k$ and $b\in A^l$
$$D(ab)=(Da)b+(-1)^k aDb$$

满足三条性质

(i) exterior differentiation $d:\Omega^{\ast}(U)->\Omega^{\ast}(U)$ is antiderivation of degree 1:

$$d(\omega \wedge \tau)=(d\omega)\wedge \tau+(-1)^{deg\space \omega}\omega\wedge d \tau$$

(ii) $d^2=0$ ;  
(iii) If $f\in C^{\infty}(U)$ and $X\in \mathfrak{X}(U)$, then $(df)(X)=Xf.$

这三条性质唯一确定了外微分。

## Applications to Vector Calculus

函数 $F=\langle P,Q,R \rangle:U\rightarrow \mathbb{R}^3$

$F_p\in \mathbb{R}^3\simeq T_p\mathbb{R}^3$

$$
\text{grad} f = \left[\begin{array}{c}
     \partial / \partial x\\
     \partial / \partial y\\
     \partial / \partial z
   \end{array}\right] f = \left[\begin{array}{c}
     f_x\\
     f_y\\
     f_z
   \end{array}\right]
$$

$$
\text{curl} \left[\begin{array}{c}
     P\\
     Q\\
     R
   \end{array}\right] = \left[\begin{array}{c}
     \partial / \partial x\\
     \partial / \partial y\\
     \partial / \partial z
   \end{array}\right] \times \left[\begin{array}{c}
     P\\
     Q\\
     R
   \end{array}\right] = \left[\begin{array}{c}
     R_y - Q_z\\
     - (R_x - P_z)\\
     Q_x - P_y
   \end{array}\right]
$$

$$
\text{div} \left[\begin{array}{c}
     P\\
     Q\\
     R
   \end{array}\right] = \left[\begin{array}{c}
     \partial / \partial x\\
     \partial / \partial y\\
     \partial / \partial z
   \end{array}\right] \cdot \left[\begin{array}{c}
     P\\
     Q\\
     R
   \end{array}\right] = P_x + Q_y + R_z
$$

在 $\mathbb{R}^3$ 的开子集 $U$ 上，有以下关系

<div style="text-align:center">
<script type="text/tikz" data-tex-packages='{ "amsmath": "", "amssymb": "", "amsfonts": "", "tikz-cd": "" }'>
  \begin{tikzcd}
    \Omega^0(U) \arrow[r, "d"] \arrow[d, "\simeq"] &
    \Omega^1(U) \arrow[r, "d"] \arrow[d, "\simeq"] &
    \Omega^2(U) \arrow[r, "d"] \arrow[d, "\simeq"] &
    \Omega^3(U) \arrow[d, "\simeq"] \\
    C^{\infty}(U) \arrow[r, "\operatorname{grad}"] &
    \mathcal{X}(U) \arrow[r, "\operatorname{curl}"] &
    \mathcal{X}(U) \arrow[r, "\operatorname{div}"] &
    C^{\infty}(U)
  \end{tikzcd}
</script>
</div>

tikz-cd 适合绘制这种 commutative diagram，但是 mathjax 对这个的支持不是很好，最近ai-folio模板更新了 v1.0 版本，有人修复了 tikzjax 的问题，更新到了 v1.0 版本后又费了挺多token终于跑通了。

<!-- ![deRham complex](/assets/img/posts/diagram.svg) -->

<!-- ![deRham complex](/assets/img/posts/diagram_from_pdf.svg){: style="width:30%; display:block; margin:auto"} -->
