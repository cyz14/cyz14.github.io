---
layout: post
title: The Tangent Space, Submanifolds, Categories and Functors
date: 2026-06-08 21:25:00
description: notes of Chap 3 of The Introduction to Manifolds, Loring W. Tu
tags: differential geometry
categories: math
tikzjax: true
---

<body>
    <p>
      根据定义，流形上一点的切空间是这点导数的向量空间。
      流形的一个光滑映射诱导了一个线性映射，叫作它的微分
      <i>differential</i>，或在那些点的切空间。在局部坐标系，微分用这个映射的偏导数的Jacobian矩阵表示。在这个意义下，流形间映射的微分是欧氏空间之间映射的导数的推广。
    </p>
    <p>
      流形理论中的一个基本原理是线性化原理，一个流形在一点附近可以用它的切空间近似，一个光滑映射可以用这个映射的微分近似。这个意义下，我们把拓扑问题转为一个线性问题。线性化的一个好例子是反函数定理，把一个光滑映射的局部可逆性转为它在一点的微分的可逆性。
    </p>
    <p>
      利用微分，我们把一点处具有最大秩的映射分为
      immersions 和
      submersions，根据这个微分在这点是否是单射和满射。微分是满射的点是映射的一个
      <i>regular point</i>。Regular level set theorem
      说一个所有点是 regular
      的 level set 是一个 regular
      submanifold，即，一个子集，局部像一个
      \(\mathbb{R}^n\) 中的 \(k\)-plane.
      这个定理提供了一个证明一个拓扑空间是流形的有力工具。
    </p>
    <p>
      我们然后介绍范畴
      categories 和函子
      functors，一个比较结构性相似的框架。我们再回到用微分研究映射。根据微分的秩，我们可以得到光滑映射的三个局部法向形式&ndash;常数秩定理，immersion
      定理和 submersion
      定理，分别对应于常数秩微分，单射微分和满射微分。
    </p>
    <p>
      流形切空间的整体可以给一个向量丛
      <i>vector bundle</i>
      的结构，被叫作流形的切丛
      <i>tangent
      bundle</i>。直觉来说，一个流形上的向量丛是用流形上的点来参数化的一族局部平凡的向量空间。流形间的一个光滑映射通过它每点的微分诱导了对应切丛间的一个丛映射。这样我们得到一个从光滑流形和光滑映射范畴到向量从和丛映射的范畴的协变函子
      covariant
      functor。向量场，在物理世界中体现为速度、力、电、磁等等，可以被视为流形上切丛的截面。
    </p>
    <p>
      光滑 \(C^{\infty}\) bump
      函数和单位分解是光滑流形理论中的重要技术工具。基于
      \(C^{\infty}\) bump
      函数我们给出几个向量场是光滑的判断条件。这一章以积分曲线，流和光滑向量场的李括号结束。
    </p>
    <h2 id="auto-1">1<span style="margin-left: 1em"></span>The Tangent Space
    切空间<span style="margin-left: 1em"></span></h2>
    <p>
      \(\mathbb{R}^n\) 中开集 \(U\)
      中任意点 \(p\)
      有两个等价方式定义一个切向量：
    </p>
    <ol>
      <li>
        <p>
          <b>一个箭头，表示为一个列向量</b>；
        </p>
      </li>
      <li>
        <p>
          一个点的\(C^{\infty}_p\)导数，点\(p\)
        </p>
      </li>
    </ol>
    <p>
      两种方式都可以被拓展到流形上。第一种直观，但是处理复杂，比如在两个图册中找到同一个箭头。
    </p>
    <p>
      最清晰和内蕴的方式是作为一个点导数，也是这本书选择的方法。
    </p>
    <h3 id="auto-2">1.1<span style="margin-left: 1em"></span>The Tangent Space at a Point<span style="margin-left: 1em"></span></h3>
    <p>
      和 \(\mathbb{R}^n\)
      中一样，我们定义 \(M\)
      中点 \(p\) 处 \(C^{\infty}\)
      函数的芽 germ 为定义在
      \(p\) 在 \(M\) 中的邻域上的
      \(C^{\infty}\)
      函数的等价类。\(C^{\infty}\)实值函数的芽记为
      \(C_p^{\infty}
      (M)\)。函数的加法和乘法让它构成一个环；附上实数的标量乘法，\(C_p^{\infty}
      (M)\) 变成 \(\mathbb{R}\)
      上的一个代数。
    </p>
    <p>
      扩展 \(\mathbb{R}^n\)
      中点处的导数，我们定义流形
      \(M\) 上一点的导数，或
      \(C_p^{\infty} (M)\)
      的点导数，为一个线性映射
      \(D : C_p^{\infty} (M) \rightarrow \mathbb{R}\) 使得
    </p>
    <center>
      \(\displaystyle D (f g) = (D f) g (p) + f (p) D g.\)
    </center>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">1.1</class>.
      </strong><i>流形 \(M\) 中点 \(p\)
      的一个切向量是点 \(p\)
      的一个导数。</i>
    </p>
    <p>
      和 \(\mathbb{R}^n\) 中一样，点\(p\)
      的切空间构成一个向量空间
      \(T_p
      (M)\)，叫作切空间。也写作
      \(T_p M\)。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">1.2</class>. </strong>(Tangent
      space to an open subset) 如果 \(U\) 是 \(M\)
      中一个包含点 \(p\)
      的开集，那么 \(C^{\infty}\)
      函数芽在 \(U\) 中 \(p\)
      的代数 \(C_p^{\infty} (U)\) 和 \(C_p^{\infty}
      (M)\) 一样。因此，\(T_p U = T_p M\).
    </p>
    <p>
      给定一个坐标邻域 \((U,
      \phi) = (U, x^1, \ldots,
      x^n)\)。重温偏导数 \(\partial /
      \partial x^i\) 的定义。令 \(r^1, \ldots,
      r^n\) 为 \(\mathbb{R}^n\)
      中的标准坐标。则
    </p>
    <center>
      \(\displaystyle x^i = r^i \circ \phi : U \rightarrow \mathbb{R}.\)
    </center>
    <p>
      如果 \(f\) 是一个 \(p\)
      邻域的光滑函数，我们设
    </p>
    <center>
      \(\displaystyle \frac{\partial}{\partial x^i} |_p f =
      \frac{\partial}{\partial r^i} |_{\phi
(p)} (f \circ \phi^{- 1}) \in
      \mathbb{R}.\)
    </center>
    <p>
      容易检查 \(\partial / \partial x^i |_p\)
      满足导数性质所以是一个
      \(p\) 点的切向量。
    </p>
    <p>
      当 \(M\) 是一维且 \(t\)
      是一个局部坐标，习惯把
      \(p\) 点的坐标向量写成
      \(d / d t|_p\) 而不是 \(\partial / \partial
      t|_p\)。为了简化记号，我们有时会写
      \(\partial / \partial x^i\) 而不是 \(\partial /
      \partial x^i |_p\)
      如果知道这个切向量在哪里。
    </p>
    <h3 id="auto-3">1.2<span style="margin-left: 1em"></span>The Differential of a Map<span style="margin-left: 1em"></span></h3>
    <p>
      Let \(F : N \rightarrow M\) be a \(C^{\infty}\) map between two
      manifolds. At each point \(p \in N\), the map \(F\) induces a linear map
      of tangent spaces, called its <i>differential</i> at \(p\),
    </p>
    <center>
      \(\displaystyle F_{\ast} : T_p N \rightarrow T_{F (p)} M\)
    </center>
    <p>
      as follows. If \(X_p \in T_p N\), then \(F_{\ast} (X_p)\) is the tangent
      vector in \(T_{F (p)} M\) defined by
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle (F_{\ast} (X_p)) f = X_p (f \circ F)
        \in \mathbb{R} \quad \operatorname{for}f
\in C_{F (p)}^{\infty}
        (M)\)</td>
        <td align="right">(1.1)</td>
      </tr>
    </table>
    <p>
      Here \(f\) is a germ at \(F (p)\), represented by a \(C^{\infty}\)
      function in a neighborhood of \(F (p)\).
      上式跟芽的表示无关，所以我们实际中可以忽视一个芽和它的代表函数之间的区别。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.3</class>. </strong>(The differential
      of a map). Check that \(F_{\ast} (X_p)\) is a derivation at \(F (p)\)
      and that \(F_{\ast} : T_p N \rightarrow T_{F (p)} M\) is a linear map.
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">1.4</class>. </strong>(Differential of a
      map between Euclidean spaces). 
    </p>
    <p>
      Suppose \(F : \mathbb{R}^n \rightarrow \mathbb{R}^m\) is smooth and
      \(p\) is a point in \(\mathbb{R}^n\).
    </p>
    <p>
      \(x^1, \ldots, x^n\) be the coordinates on \(\mathbb{R}^n\), \(y^1,
      \ldots, y^m\) the coordinates on \(\mathbb{R}^m\). 
    </p>
    <p>
      Tangent vectors \(\partial / \partial x^1 |_p, \ldots, \partial /
      \partial x^n |_p\) form a basis for the vector space \(T_p
      (\mathbb{R}^n)\).
    </p>
    <p>
      \(\partial / \partial y^1 |_{F (p)}, \ldots, \partial / \partial y^m
      |_{F (p)}\) form a basis for the tangent space \(T_{F (p)}
      (\mathbb{R}^m)\).
    </p>
    <p>
      The linear map \(F_{\ast} : T_p (\mathbb{R}^n) \rightarrow T_{F (p)}
      (\mathbb{R}^m)\) is described by a matrix \([a_j^i]\) relative to these
      two bases:
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle F_{\ast} \left(
        \frac{\partial}{\partial x^j} |_p \right) = \sum_k
        a^k_j
\frac{\partial}{\partial y^k} |_{F (p)}, \quad a^k_j \in
        \mathbb{R}.\)</td>
        <td align="right">(1.2)</td>
      </tr>
    </table>
    <p>
      Let \(F^i = y^i \circ F\) be the \(i\)th component of \(F\). We can find
      \(a^i_j\) by evaluating the right hand side (RHS) and the left-hand-side
      (LHS) of above on \(y^i\):
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  \operatorname{RHS}= & \sum_k a^k_j
      \frac{\partial}{\partial y^k} |_{F (p)}
  y^i = \sum_k a^k_j \delta^i_k
      = a^i_j,\\
  \operatorname{LHS}= & F_{\ast} \left(
      \frac{\partial}{\partial x^j} |_p
  \right) y^i =
      \frac{\partial}{\partial x^j} |_p (y^i \circ F) =
  \frac{\partial
      F^i}{\partial x^j} (p) .
\end{array}\)
    </center>
    <p style="margin-bottom: 1em">
      So the matrix of \(F_{\ast}\) relative to the bases \(\{ \partial /
      \partial x^j |_p \}\) and \(\{ \partial / \partial y^i |_{F (p)} \}\) is
      \([\partial F^i / \partial x^j (p)]\). This is precisely the Jacobian
      matrix of the derivatives of \(F\) at \(p\). Thus, the differential of a
      map between manifolds generalizes the derivative of a map between
      Euclidean spaces.
    </p>
    <h3 id="auto-4">1.3<span style="margin-left: 1em"></span>The Chain Rule<span style="margin-left: 1em"></span></h3>
    <p>
      Let \(F : N \rightarrow M\) and \(G : M \rightarrow P\) be smooth maps
      of manifolds, and \(p \in N\). The differentials of \(F\) at \(p\) and
      \(G\) at \(F (p)\) are linear maps:
    </p>
    <center>
      \(\displaystyle T_p N \xrightarrow{F_{\ast, p}} T_{F (p)} M
      \xrightarrow{G_{\ast, F (p)}} T_{G
(F (p))} P.\)
    </center>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">1.5</class>. </strong><i>(The
      chain rule). If \(F : N - M\) and \(G : M \rightarrow P\) are smooth
      maps of manifolds and \(p \in N\). Then</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle (G \circ F)_{\ast, p} = G_{\ast, F (p)} \circ F_{\ast,
        p} .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>注意. </strong>The differential of the identity map
      \(\mathbb{1 }_M : M \rightarrow M\) at any point \(p\) in \(M\) is the
      identity map
    </p>
    <center>
      \(\displaystyle \mathbb{1}_{T_p M} : T_p M \rightarrow T_p M,\)
    </center>
    <p>
      because 
    </p>
    <center>
      \(\displaystyle ((\mathbb{1}_M)_{\ast} X_p) f = X_p (f \circ
      \mathbb{1}_M) = X_p f,\)
    </center>
    <p style="margin-bottom: 1em">
      for any \(X_p \in T_p M\) and \(f \in C_p^{\infty} (M)\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>推论 <class style="font-style: normal">1.6</class>.
      </strong><i>\(F_{\ast} : T_p N \rightarrow T_{F (p)} M\) is a
      isomorphism of vector spaces.</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>推论 <class style="font-style: normal">1.7</class>.
      </strong><i>(Invariance of dimension). If an open set \(U \subset
      \mathbb{R}^n\) is diffeomorphic to an open set \(V \subset
      \mathbb{R}^m\), then \(n = m\).</i>
    </p>
    <h3 id="auto-5">1.4<span style="margin-left: 1em"></span>Bases for the Tangent Space at a Point<span
    style="margin-left: 1em"></span></h3>
    <p>
      Denote by \(r^1, \ldots, r^n\) the standard coordinates on
      \(\mathbb{R}^n\), and if \((U, \phi)\) is a chart about a point \(p\) in
      a manifold \(M\) of dimension \(n\), we set \(x^i = r^i \circ \phi\).
      Since \(\phi : U \rightarrow \mathbb{R}^n\) is a diffeomorphism onto its
      image, by corollary above, the differential 
    </p>
    <center>
      \(\displaystyle \phi_{\ast} : T_p M \rightarrow T_{\phi (p)}
      \mathbb{R}^n\)
    </center>
    <p>
      is a vector space isomorphism. The tangent space \(T_p M\) has the same
      dimension \(n\) as the manifold \(M\).
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.8</class>. </strong><i>Let
      \((U, \phi) = (U, x^1, \ldots, x^n)\) be a chart about a point \(p\) in
      a manifold \(M\). Then</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \phi_{\ast} \left( \frac{\partial}{\partial x^i} |_p
        \right) =
\frac{\partial}{\partial r^i} |_{\phi (p)} .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.9</class>. </strong><i>If \((U,
      \phi) = (U, x^1, \ldots, x^n)\) is a chart containing \(p\), then the
      tangent space \(T_p M\) has basis</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \frac{\partial}{\partial x^1} |_p, \ldots,
        \frac{\partial}{\partial x^n} |_p .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.10</class>.
      </strong><i>(Transition matrix for coordinate vectors). Suppose \((U,
      x^1, \ldots, x^n)\) and \((V, y^1, \ldots, y^n)\) are two coordinates
      charts on a manifold \(M\). Then</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \frac{\partial}{\partial x^j} = \sum_i \frac{\partial
        y^i}{\partial x^j}
\frac{\partial}{\partial y^i}\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>on \(U \cap V\).</i>
    </p>
    <h3 id="auto-6">1.5<span style="margin-left: 1em"></span>A Local Expression for the
    Differential<span style="margin-left: 1em"></span></h3>
    <p>
      Given a smooth map \(F : N \rightarrow M\) of manifolds and \(p \in N\),
      let \((U, x^1, \ldots, x^n)\) be a chart about \(p\) in \(N\) and let
      \((V, y^1, \ldots, y^m)\) be a chart about \(F (p)\) in \(M\). We will
      find a local expression for the differential \(F_{\ast, p} : T_p N
      \rightarrow T_{F (p)} M\) relative to the two charts.
    </p>
    <p>
      By proposition 9, \(\{ \partial / \partial x^j |_p \}_{j = 1}^n\) is a
      basis for \(T_p N\) and \(\{ \partial / \partial y^i |_{F (p)} \}_{i =
      1}^m\) is a basis for \(T_{F (p)} M\). Therefore, the differential
      \(F_{\ast} = F_{\ast, p}\) is completely determined by the numbers
      \(a_j^i\) such that
    </p>
    <center>
      \(\displaystyle F_{\ast} \left( \frac{\partial}{\partial x^j} |_p
      \right) = \sum_{k = 1}^m
a_j^k \frac{\partial}{\partial y^k} |_{F (p)},
      \quad j = 1, \ldots, n.\)
    </center>
    <p>
      Applying both sides to \(y^i\), we find that
    </p>
    <center>
      \(\displaystyle a^i_j = \left( \sum_{k = 1}^m a_j^k
      \frac{\partial}{\partial y^k} |_{F (p)}
\right) y^i = F_{\ast} \left(
      \frac{\partial}{\partial x^j} |_p \right) y^i =
\frac{\partial}{\partial
      x^j} |_p (y^i \circ F) = \frac{\partial F^i}{\partial
x^j} (p) .\)
    </center>
    <p>
      We state this result as a proposition.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1.11</class>. </strong><i>Given a
      smooth map \(F : N - M\) of manifolds and a point \(p \in N\), let \((U,
      x^1, \ldots, x^n)\) and \((V, y^1, \ldots, y^m)\) be coordinate charts
      about \(p\) in \(N\) and \(F (p)\) in \(M\) respectively. Relative to
      the bases \(\{ \partial / \partial x^j |_p \}\) for \(T_p N\) and \(\{
      \partial / \partial y^i |_{F (p)} \}\) for \(T_{F (p)} M\), the
      differential \(F_{\ast, p} : T_p N \rightarrow T_{F (p)} M\) is
      represented by the matrix \([\partial F^i / \partial x^j (p)]\), where
      \(F^i = y^i \circ F\) is the \(i\)th component of \(F\).</i>
    </p>
    <p>
      This proposition is in the spirit of the &quot;arrow&quot; approach to
      tangent vectors.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">1.12</class>. </strong>(Inverse
      function theorem). The inverse function theorem for manifolds has a
      coordinate-free description: a \(C^{\infty}\) map \(F : N \rightarrow
      M\) between two manifolds of the same dimension is locally invertible if
      and only if its differential \(F_{\ast, p} : T_p N \rightarrow T_{F (p)}
      M\) at \(p\) is an isomorphism.
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">1.13</class>. </strong>(The chain rule in
      calculus notation). Suppose \(w = G (x, y, z)\) is a \(C^{\infty}\)
      function: \(\mathbb{R}^3 \rightarrow \mathbb{R}\) and \((x, y, z) = F
      (t)\) is a \(C^{\infty}\) function: \(\mathbb{R}  \rightarrow
      \mathbb{R}^3\). Under composition, 
    </p>
    <center>
      \(\displaystyle w = (G \circ F) (t) = G (x (t), y (t), z (t))\)
    </center>
    <p>
      becomes a \(C^{\infty}\) function of \(t \in \mathbb{R}\). The
      differentials \(F_{\ast}, G_{\ast}\), and \((G \circ F)_{\ast}\) are
      respresented by the matrices
    </p>
    <center>
      \(\displaystyle \begin{array}{ccc}
  \left[ \begin{array}{c}
    d x / d
      t\\
    d y / d t\\
    d z / d t
  \end{array} \right], & \left[
      \begin{array}{ccc}
    \frac{\partial w}{\partial x} & \frac{\partial
      w}{\partial y} &
    \frac{\partial w}{\partial z}
  \end{array}
      \right], & \frac{d w}{d t}
\end{array}\)
    </center>
    <p>
      respectively. Composition of linear maps is represented by matrix
      multiplication, in terms of the chain rule \((G \circ F)_{\ast} =
      G_{\ast} \circ F_{\ast}\), is equivalent to 
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle \frac{d w}{d t} = \left[ \begin{array}{ccc}
 
        \frac{\partial w}{\partial x} & \frac{\partial w}{\partial y} &
 
        \frac{\partial w}{\partial z}
\end{array} \right] \left[
        \begin{array}{c}
  d x / d t\\
  d y / d t\\
  d z / d t
\end{array}
        \right] = \frac{\partial w}{\partial x} \frac{d x}{d t}
        +
\frac{\partial w}{\partial y} \frac{d y}{d t} + \frac{\partial
        w}{\partial z}
\frac{d z}{d t} .\)
      </center>
    </p>
    <h3 id="auto-7">1.6<span style="margin-left: 1em"></span>Curves in a Manifold<span style="margin-left: 1em"></span></h3>
    <p>
      A <i>smooth curve</i> in a manifold is by definition a smooth map \(c :]
      a, b [\rightarrow M\) from some open interval into \(M\). Usually we
      assume \(0 \in] a, b [\) and say that \(c\) is a curve starting at \(p\)
      if \(c (0) = p\). The velocity vector \(c' (t_0)\) of the curve \(c\) at
      time \(t_0 \in] a, b [\) is defined to be 
    </p>
    <center>
      \(\displaystyle c' (t_0) := c_{\ast} \left( \frac{d}{d t} |_{t_0}
      \right) \in T_{c (t_0)} M.\)
    </center>
    <p>
      Alternative notation for \(c' (t_0)\) are 
    </p>
    <center>
      \(\displaystyle \frac{d c}{d t} (t_0) \quad \operatorname{and} \quad
      \frac{d}{d t} |_{t_0} c.\)
    </center>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意. </strong>When \(c :] a, b [\rightarrow
      \mathbb{R}\) is a curve with target space \(\mathbb{R}\), the notation
      \(c' (t)\) can be a source of confusion. Here \(t\) is a standard
      coordinate on the domain \(] a, b [\). Let \(x\) be the standard
      coordinate on the target space \(\mathbb{R}\). By our definition, \(c'
      (t)\) is a tangent vector at \(c (t)\), hence a multiple of \(d / d
      x|_{c (t)}\). On the other hand, in calculus, notation \(c' (t)\) is the
      derivative of a real-valued function and is therefore a scalar. If it is
      necessary to distinguish between these two meanings of \(c' (t)\) when
      \(c\) maps into \(\mathbb{R}\), we will write \(\dot{c} (t)\) for the
      calculus derivative.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.14</class>. </strong>(Velocity vector
      versus the calculus derivative). Verify that \(c' (t) = \dot{c} (t) d /
      d x|_{c (t)}\).
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.15</class>.
      </strong><i>(Velocity of a curve in local coordinates). Let \(c :] a, b
      [\rightarrow M\) be a smooth curve, and let \((U, x^1, \ldots, x^n)\) be
      a coordinate chart about \(c (t)\). Write \(c^i = x^i \circ c\) for the
      \(i\)th component of \(c\) in the chart. Then \(c' (t)\) is given by</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle c' (t) = \sum_{i = 1}^n \dot{c} (t)
        \frac{\partial}{\partial x^i} |_{c (t)} .\)
      </center></i>
    </p>
    <p>
      <i>Thus, relative to the basis \(\{ \partial / \partial x^i |_p \}\) for
      \(T_{c (t)} M\), the velocity \(c' (t)\) is represented by the column
      vector</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \left[ \begin{array}{c}
  \dot{c}^1 (t)\\
  \vdots\\
 
        \dot{c}^n (t)
\end{array} \right] .\)
      </center></i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1.16</class>.
      </strong><i>(Existence of a curve with a given initial vector). For any
      point \(p\) in a manifold \(M\) and any tangent vector \(X_p \in T_p
      M\), there are \(\varepsilon > 0\) and a smooth curve \(c :] -
      \varepsilon, \varepsilon [\rightarrow M\) such that \(c (0) = p\) and
      \(c' (0) = X_p\).</i>
    </p>
    <p>
      In definition 1, we defined a tangent vector at a point \(p\) of a
      manifold abstractly as a derivation at \(p\). Using curves, we can now
      interpret a tangent vector geometrically as a directional derivative.
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.17</class>. </strong><i>Suppose
      \(X_p\) is a tangent vector at a point \(p\) of a manifold \(M\) and \(f
      \in C_p^{\infty} (M)\). If \(c :] - \varepsilon, \varepsilon
      [\rightarrow M\) is a smooth curve starting at \(p\) and \(c' (0) =
      X_p\), then</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle X_p f = \frac{d}{d t} |_0 (f \circ c) .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>证明. </strong>By the definition of \(c' (0)\) and
      \(c_{\ast}\),
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle X_p f = c' (0) f = c_{\ast} \left( \frac{d}{d t} |_0
        \right) f = \frac{d}{d t}
|_0 (f \circ c) .\)
      </center>
      <span style="margin-left: 1em"></span>
      \(\Box\)
    </p>
    <h3 id="auto-8">1.7<span style="margin-left: 1em"></span>Computing the Differential Using
    Curves<span style="margin-left: 1em"></span></h3>
    <p>
      前面介绍了两种计算光滑映射的微分的方法，公式1的求一点处的导数，和命题11的局部坐标方法。下个命题使用曲线计算微分
      \(F_{\ast, p}\).
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.18</class>. </strong><i>let \(F
      : N \rightarrow M\) be a smooth map of manifolds, \(p \in N\), and \(X_p
      \in T_p N\). If \(c\) is a smooth curve starting at \(p\) in \(N\) with
      velocity \(X_p\) at \(p\), then </i>
    </p>
    <p>
      <i><center>
        \(\displaystyle F_{\ast, p} (X_p) = \frac{d}{d t} |_0 (F \circ c) (t)
        .\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>In other words, \(F_{\ast, p} (X_p)\) is the velocity vector of the
      image curve \(F \circ c\) at \(F (p)\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.19</class>. </strong>(Differential of
      left multiplication). If \(g\) is a matrix in the general linear group
      \(\operatorname{GL} (n, \mathbb{R})\), let \(\ell_g : \operatorname{GL}
      (n, \mathbb{R}) \rightarrow \operatorname{GL} (n,
\mathbb{R})\) be left
      multiplication by \(g\); thus, \(\ell_g (B) = g B\) for any \(B \in
      \operatorname{GL} (n, \mathbb{R})\). Since \(\operatorname{GL} (n,
      \mathbb{R})\) is an open subset of the vector space \(\mathbb{R}^{n
      \times n}\), the tangent space \(T_g (\operatorname{GL} (n,
      \mathbb{R}))\) can be identified with \(\mathbb{R}^{n \times n}\). Show
      that with this identification the differential \((\ell_g)_{\ast, I} :
      T_I (\operatorname{GL} (n, \mathbb{R})) \rightarrow
      T_g
(\operatorname{GL} (n, \mathbb{R}))\) is also left multiplication by
      \(g\).
    </p>
    <p>
      Solution. Let \(X \in T_I (\operatorname{GL} (n, \mathbb{R}))
      =\mathbb{R}^{n \times n}\). To compute \((\ell_g)_{\ast, I} (X)\),
      choose a curve \(c (t)\) in \(\operatorname{GL} (n, \mathbb{R})\) with
      \(c (0) = I\) and \(c' (0) = X\). Then \(\ell_g (c (t)) = g c (t)\) is
      simply matrix multiplication. By proposition 18,
    </p>
    <center>
      \(\displaystyle (\ell_g)_{\ast, I} (X) = \frac{d}{d t} |_{t = 0} \ell_g
      (c (t)) = \frac{d}{d
t} |_{t = 0} g c (t) = g c' (0) = g X.\)
    </center>
    <h3 id="auto-9">1.8<span style="margin-left: 1em"></span>Immersions and Submersions<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">1.20</class>. </strong>Suppose
      \(N\) and \(M\) are manifolds of dimensions \(n\) and \(m\)
      respectively. Then \(\dim T_p N = n\) and \(\dim T_{F (p)} M = m\). If
      \(F : N \rightarrow M\) is an immersion at a point of \(N\), then \(n
      \leq m\) and if \(F\) is a submersion a point of \(N\), then \(n \geq
      m\).
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">1.21</class>. </strong>The prototype of
      an immersion is the inclusion of \(\mathbb{R}^n\) in a higher dimension
      \(\mathbb{R}^m\):
    </p>
    <center>
      \(\displaystyle i (x^1, \ldots, x^n) = (x^1, \ldots, x^n, 0, \ldots, 0)
      .\)
    </center>
    <p>
      The prototype of a submersion is the projection of \(\mathbb{R}^n\) onto
      a lower-dimensional \(\mathbb{R}^m\):
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle \pi (x^1, \ldots, x^m, x^{m + 1}, \ldots, x^n) = (x^1,
        \ldots, x^m) .\)
      </center>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>If U is an open subset of a manifold \(M\),
      then the inclusion \(i : U \rightarrow M\) is both an immersion and a
      submersion. This example shows in particular that a submersion need not
      be onto.
    </p>
    <p>
      In section 11 we will take a more in-depth analysis of immersions and
      submersions. According to the immersion and submersion theorems to be
      proven there, every immersion is locally an inclusion and every
      submersion is locally a projection.
    </p>
    <h3 id="auto-10">1.9<span style="margin-left: 1em"></span>Rank, and Critical and Regular Points<span
    style="margin-left: 1em"></span></h3>
    <p>
      The rank of a linear transformation \(L : V \rightarrow W\) between
      finite-dimensional vector spaces is the dimension of the image \(L (V)\)
      as a subspace of \(W\), while the rank of a matrix \(A\) is the
      dimension of its column space. If \(L\) is represented by a matrix \(A\)
      relative to a basis for \(V\) and a basis for \(W\), then the rank of
      \(L\) is the same as the rank of \(A\), because the image \(L (V)\) is
      simply the column space of \(A\).
    </p>
    <p>
      Now consider a smooth map \(F : N \rightarrow M\) of manifolds. Its rank
      at a point \(p\) in \(N\) , denoted by \(\operatorname{rk}F (p)\), is
      defined as the rank of the differential \(F_{\ast, p} : T_p N
      \rightarrow T_{F (p)} M\). Relative to the coordinate neighborhoods, the
      differential is represented by the Jacobian matrix \([\partial F^i /
      \partial x^j (p)]\), so 
    </p>
    <center>
      \(\displaystyle \operatorname{rk}F (p) =\operatorname{rk} \left[
      \frac{\partial F^i}{\partial
x^j} (p) \right] .\)
    </center>
    <p>
      Since the differential of a map is independent of coordinate charts, so
      is the rank of a Jacobian matrix.
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">1.22</class>. </strong><i>A point
      \(p\) in \(N\) is a critical point of \(F\) is the differential</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle F_{\ast, p} : T_p N \rightarrow T_{F (p)} M\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>fails to be surjective. It is a regular point of \(F\) if the
      differential \(F_{\ast, p}\) is surjective. In other words, \(p\) is a
      regular point of the map \(F\) if and only if \(F\) is a submersion at
      \(p\). A point in \(M\) is a critical value if it is the image of a
      critical point; otherwise it is a regular value.</i>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.23</class>. </strong><i>For a
      real-valued function \(f : M \rightarrow \mathbb{R}\), a point \(p\) in
      \(M\) is a critical point if and only if relative to some chart \((U,
      x^1, \ldots, x^n)\) containing \(p\), all the partial derivatives
      satisfy</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \frac{\partial f}{\partial x^j} (p) = 0, \quad j = 1,
        \ldots, n.\)
      </center></i>
    </p>
    <h2 id="auto-11">2<span style="margin-left: 1em"></span>Submanifolds<span style="margin-left: 1em"></span></h2>
    <p>
      Introduce the concept of a regular submanifold of a manifold, a subset
      that is locally defined by the vanishing of some of the coordinate
      functions. Using the inverse function theorem, we derive a criterion,
      called the regular level set theorem, that can often be used to show
      that a level set of a \(C^{\infty}\) map of manifolds is a regular
      submanifold and therefore a manifold.
    </p>
    <p>
      
    </p>
    <h3 id="auto-12">2.1<span style="margin-left: 1em"></span>Submanifolds<span style="margin-left: 1em"></span></h3>
    <p>
      The \(x y\)-plane in \(\mathbb{R}^3\) is the prototype of a regular
      submanifold of a manifold. 
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">2.1</class>.
      </strong><i>Vanishing of \(n - k\) coordinate functions.</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">2.2</class>. </strong><i>If \(S\)
      is a regular submanifold of dimension \(k\) in a manifold \(N\) of
      dimension \(n\), then \(n - k\) is said to be the codimension of \(S\)
      in \(N\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意. </strong>As a topological space, a regular
      submanifold of \(N\) is required to have the subspace topology.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">2.3</class>. </strong>\(f (x) = \sin (1 /
      x)\) and the union with open interval \(] - 1, 1 [\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">2.4</class>. </strong><i>\(S\) be
      a regular submanifold of \(N\) and \(\mathfrak{U}= \{ (U, \phi) \}\) a
      collection of compatible adapted charts of \(N\) that covers \(S\). Then
      \(\{ (U \cap S, \phi_S) \}\) is an atlas for \(S\). Therefore, a regular
      submanifold is itself a manifold.</i>
    </p>
    <h3 id="auto-13">2.2<span style="margin-left: 1em"></span>Level Set of a Function<span style="margin-left: 1em"></span></h3>
    <p>
      A level set of a map \(F : N \rightarrow M\) is a subset
    </p>
    <center>
      \(\displaystyle F^{- 1} (\{ c \}) = \{ p \in N|F (p) = c \}\)
    </center>
    <p>
      for some \(c \in M\). The usual notation for a level set if \(F^{- 1}
      (c)\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">2.5</class>. </strong>If a
      regular level set is nonempty, say \(p \in F^{- 1} (c)\), then the map
      \(F : N \rightarrow M\) is a submersion at \(p\). By remark 20, \(\dim N
      \geqslant \dim M\).
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">2.6</class>. </strong>(The 2-sphere in
      \(\mathbb{R}^3\)). The unit 2-sphere 
    </p>
    <center>
      \(\displaystyle S^2 = \{ (x, y, z) \in \mathbb{R}^3 |x^2 + y^2 + z^2 = 1
      \}\)
    </center>
    <p style="margin-bottom: 1em">
      is the level set \(g^{- 1} (1)\) of level 1 of the function \(g (x, y,
      z) = x^2 + y^2 + z^2\).
    </p>
    <p>
      We show that any regular level set \(g^{- 1} (c)\) of a \(C^{\infty}\)
      real function \(g\) on a manifold can be expressed as a regular zero
      set.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>引理 <class style="font-style: normal">2.7</class>. </strong><i>let \(g
      : N \rightarrow \mathbb{R}\) be a \(C^{\infty}\) function. A regular
      level set \(g^{- 1} (c)\) of the function \(g\) is the regular zero set
      \(f^{- 1} (0)\) of the function \(f = g - c\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">2.8</class>. </strong><i>Let \(g
      : N \rightarrow \mathbb{R}\) a \({C^{\infty}} \) function on manifold
      \(N\).The a nonempty regular level set \(S = g^{- 1} (c)\) is a regular
      submanifold of \(N\) of codimension 1.</i>
    </p>
    <h3 id="auto-14">2.3<span style="margin-left: 1em"></span>The Regular Level Set Theorem<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">2.9</class>. </strong><i>(Regular
      level set theorem). Let \(F : N \rightarrow M\) be a \(C^{\infty}\) map
      of manifolds, with \(\dim N = n\) and \(\dim M = m\). Then a nonempty
      regular level set \(F^{- 1} (c)\), where \(c \in M\), is a regular
      submanifold of \(N\) of dimension equal to \(n - m\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>引理 <class style="font-style: normal">2.10</class>. </strong><i>\(F : N
      \rightarrow \mathbb{R}^m\) be a \(C^{\infty}\) map on a manifold \(N\)
      of dimension \(n\) and let \(S\) be the level set \(F^{- 1}
      (\boldsymbol{0})\). If relative to some coordinate chart \((U, x^1,
      \ldots, x^n)\) about \(p \in S\), the Jacobian determinant \(\partial
      (F^1, \ldots, F^m) / \partial (x^{j_1}, \ldots, x^{j_m}) (p)\) is
      nonzero, then in some neighborhood of \(p\) one may replace \(x^{j_1},
      \ldots, x^{j_m}\) by \(F^1, \ldots, F^m\) to obtain an adapted chart for
      \(N\) relative to \(S\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意. </strong>The regular level set theorem givs a
      sufficient but not necessary condition for a level set to be a regular
      submanifold.
    </p>
    <h3 id="auto-15">2.4<span style="margin-left: 1em"></span>Examples of Regular Submanifolds<span
    style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">2.11</class>. </strong>(Hypersurface).
      The solution set \(S\) of \(x^3 + y^3 + z^3 = 1\) in \(\mathbb{R}^3\) is
      a manifold of dimension 2.
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">2.12</class>. </strong>(Solution set of
      two polynomial equations). Decide whether the subset \(S\) of
      \(\mathbb{R}^3\) defined by the two equations
    </p>
    <center>
      \(\displaystyle x^3 + y^3 + z^3 = 1,\)
    </center>
    <center>
      \(\displaystyle x + y + z = 0\)
    </center>
    <p style="margin-bottom: 1em">
      is a regular submanifold of \(\mathbb{R}^3\).
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">2.13</class>. </strong>(Special linear
      group) As a set, the special linear group \(\operatorname{SL} (n,
      \mathbb{R})\) is the subset of \(\operatorname{GL} (n, \mathbb{R})\)
      consisting of matrices of determinant 1. Since
    </p>
    <center>
      \(\displaystyle \det (A B) = (\det A) (\det B) \)
    </center>
    <p>
      and
    </p>
    <center>
      \(\displaystyle \det (A^{- 1}) = \frac{1}{\det A},\)
    </center>
    <p>
      \(\operatorname{SL} (n, \mathbb{R})\) is a subgroup of
      \(\operatorname{GL} (n, \mathbb{R})\). To show that it is a regular
      submanifold, we let \(f : \operatorname{GL} (n, \mathbb{R}) \rightarrow
      \mathbb{R}\) be the determinant map \(f (A) = \det A\), and apply the
      regular level set theorem to \(f^{- 1} (1) =\operatorname{SL} (n,
      \mathbb{R})\). We need to check that 1 is a regular value of \(f\). Skip
      some steps.
    </p>
    <p>
      By the regular level set theorem, \(\operatorname{SL} (n, \mathbb{R})\)
      is a regular submanifold of \(\operatorname{GL} (n, \mathbb{R})\) of
      codimension 1; i.e.
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle \dim \operatorname{SL} (n, \mathbb{R}) = \dim
        \operatorname{GL} (n,
\mathbb{R}) - 1 = n^2 - 1.\)
      </center>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>Complex special linear group
      \(\operatorname{SL} (n, \mathbb{C})\) is the subgroup of
      \(\operatorname{GL} (n, \mathbb{C})\) consisting of \(n \times n\)
      complex matrices of determinant 1.
    </p>
    <h2 id="auto-16">3<span style="margin-left: 1em"></span>Categories and Functors<span style="margin-left: 1em"></span></h2>
    <p>
      数学中很多问题有共同的特征。拓扑中关心拓扑空间是否同胚
      homeomorphic，群论中关心群是否同构
      isomorphic。这带来了范畴和函子。
    </p>
    <p>
      范畴是一组对象和对象间的箭头。这些箭头，叫作
      morphisms
      ，满足映射的抽象性质，通常是保持结构的映射。光滑流形和光滑映射，向量空间和线性映射。一个函子从一个范畴到另一个范畴保持恒同
      morphism和复合 morphism。
    </p>
    <p>
      代数拓扑的很大部分是函子的研究，比如，homology，cohomology
      和 homotopy
      functors。为了函子有用，通常需要简单可计算，也要够复杂保持原来范畴重要的特征。对于光滑流形，这一平衡在
      de Rham cohomology
      中达成了。这本书后面介绍光滑流形的其他函子，比如切丛和微分形式，最终到达
      de Rham cohomology.
    </p>
    <p>
      这一章节，在定义范畴和函子后，研究向量空间的对偶作为一个函子的非平凡例子。
    </p>
    <h3 id="auto-17">3.1<span style="margin-left: 1em"></span>Categories<span style="margin-left: 1em"></span></h3>
    <p>
      A category consists of a collection of objects, and for any two objects
      \(A\) and \(B\), a set \(\operatorname{Mor} (A, B)\) of elements, called
      morphisms from \(A\) to \(B\), such that given any morphism \(f \in
      \operatorname{Mor} (A, B)\) and any morphism \(g \in \operatorname{Mor}
      (B, C)\), the composite \(g \circ f \in \operatorname{Mor} (A, C)\) is
      defined. Furthermore, the composition of morphisms is required to
      satisfy two properties:
    </p>
    <ol>
      <li>
        <p>
          the identity axiom: for each object \(A\), there is an identity
          morphism \(\mathbb{1}_A \in \operatorname{Mor} (A, A)\) such that
          for any \(f \in \operatorname{Mor} (A, B)\) and \(g \in
          \operatorname{Mor} (B, A)\),
        </p>
        <center>
          \(\displaystyle f \circ \mathbb{1}_A = f \quad \operatorname{and}
          \quad \mathbb{1}_A \circ g =
g ;\)
        </center>
      </li>
      <li>
        <p>
          the associative axiom: for \(f \in \operatorname{Mor} (A, B)\), \(g
          \in \operatorname{Mor} (B, C)\), and \(h \in \operatorname{Mor} (C,
          D)\)
        </p>
        <center>
          \(\displaystyle h \circ (g \circ f) = (h \circ g) \circ f.\)
        </center>
      </li>
    </ol>
    <p>
      If \(f \in \operatorname{Mor} (A, B)\), we often write \(f : A
      \rightarrow B\).
    </p>
    <p style="margin-top: 1em">
      <strong>例. </strong>
    </p>
    <p>
      Groups, group homeomorphisms. 
    </p>
    <p>
      Vector spaces over \(\mathbb{R}\) and \(\mathbb{R}\)-linear maps. 
    </p>
    <p>
      Topological spaces with continuous maps, called the <i>continuous
      category</i>.
    </p>
    <p>
      Smooth manifolds, smooth maps, called the <i>smooth category</i>.
    </p>
    <p style="margin-bottom: 1em">
      A pair \((M, q)\), where \(q\) is a point in manifold \(M\), called a
      pointed manifold. Given two such pairs \((N, p)\) and \((M, q)\), let
      \(\operatorname{Mor} ((N, p), (M, q))\) be the set of all smooth maps
      \(F : N \rightarrow M\) such that \(F (p) = q\). This gives rise to the
      category of pointed manifolds.
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">3.1</class>. </strong><i>Two
      objects \(A\) and \(B\) in a category are said to be isomorphic if there
      are morphisms \(f : A \rightarrow B\) and \(g : B \rightarrow A\) such
      that</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle g \circ f = \mathbb{1}_A \quad \operatorname{and}
        \quad f \circ g =
\mathbb{1}_B .\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>In this case, both \(f\) and \(g\) are called isomorphisms.</i>
    </p>
    <p>
      Usual notation for an isomorphism is &quot;\(\simeq\)&quot;.
    </p>
    <p>
      
    </p>
    <h3 id="auto-18">3.2<span style="margin-left: 1em"></span>Functors<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">3.2</class>. </strong><i>A
      (covariant) functor \(\mathcal{F}\) from one category \(\mathcal{C}\) to
      another category \(\mathcal{D}\) is a map that associates to each object
      \(A\) in \(\mathcal{C}\) an object \(\mathcal{F} (A)\) in
      \(\mathcal{D}\) and to eah morphism \(f : A \rightarrow B\) a morphism
      \(\mathcal{F} (f) : \mathcal{F} (A) \rightarrow \mathcal{F} (B)\) such
      that</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            \(\mathcal{F} (\mathbb{1}_A) = \mathbb{1}_{\mathcal{F} (A)}\),
          </p>
        </li>
        <li>
          <p>
            \(\mathcal{F} (f \circ g) =\mathcal{F} (f) \circ \mathcal{F}
            (g)\).
          </p>
        </li>
      </ol></i>
    </div>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">3.3</class>. </strong><i>Let
      \(\mathcal{F}: \mathcal{C} \rightarrow \mathcal{D}\) be a functor from a
      category \(\mathcal{C}\) to a category \(\mathcal{D}\). If \(f : A
      \rightarrow B\) is an isomorphism in \(\mathcal{C}\), then \(\mathcal{F}
      (f) : \mathcal{F} (A) \rightarrow \mathcal{F} (B)\) is an isomorphism in
      \(\mathcal{D}\).</i>
    </p>
    <p>
      If in the definition of a covariant functor we reverse the direction of
      the arrow for the morphism \(\mathcal{F} (f)\), then we obtain a
      contravariant functor. 
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">3.4</class>. </strong><i>A
      contravariant functor \(\mathcal{F}\) from one category \(\mathcal{C}\)
      to another category \(\mathcal{D}\) is a map that associates to each
      object \(A\) in \(\mathcal{C}\) an object \(\mathcal{F} (A)\) in
      \(\mathcal{D}\) and to each morphism \(f : A \rightarrow B\) a morphism
      \(\mathcal{F} (f) : \mathcal{F} (B) \rightarrow \mathcal{F} (A)\) such
      that</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            \(\mathcal{F} (\mathbb{1}_A) = \mathbb{1}_{\mathcal{F} (A)}\);
          </p>
        </li>
        <li>
          <p>
            \(\mathcal{F} (f \circ g) =\mathcal{F} (g) \circ \mathcal{F}
            (f)\).
          </p>
        </li>
      </ol></i>
    </div>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong> Smooth functions on a manifold give rise to
      a contravariant functor that associate to each manifold \(M\) the
      algebra \(\mathcal{F} (M) = C^{\infty} (M)\) of \(C^{\infty}\) functions
      on \(M\) and to each smooth map \(F : N \rightarrow M\) of manifolds the
      pullback map \(\mathcal{F} (F) = F^{\ast} : C^{\infty} (M) \rightarrow
      C^{\infty} (N)\), \(F^{\ast} (h) = h \circ F\) for \(h \in C^{\infty}
      (M)\). It is easy to verify that the pullback satisfies the two
      functorial properties.
    </p>
    <p>
      Another example of a contravariant functor is the dual of a vector
      space.
    </p>
    <h3 id="auto-19">3.3<span style="margin-left: 1em"></span>The Dual Functor and the Multicovector
    Functor<span style="margin-left: 1em"></span></h3>
    <p>
      Let \(V\) be a real vector space. Recall that its dual space
      \(V^{\vee}\) is the vector space of all linear functionals on \(V\),
      i.e., linear functions \(\alpha : V \rightarrow \mathbb{R}\). We also
      have
    </p>
    <center>
      \(\displaystyle V^{\vee} =\operatorname{Hom} (V, \mathbb{R}) .\)
    </center>
    <p>
      If \(V\) is a finite-dimensional vector space with basis \(\{ e_1,
      \ldots, e_n \}\), then its dual space has a basis the collection of
      linear functionals \(\{ \alpha^1, \ldots, \alpha^n \}\) defined as
    </p>
    <center>
      \(\displaystyle \alpha^i (e_j) = \delta^i_j, \quad 1 \leq i, j \leq n.\)
    </center>
    <p>
      A linear map \(L : V \rightarrow W\) of vector spaces induces a linear
      map \(L^{\vee}\) called the dual of \(L\), as follows. To every linear
      functional \(\alpha : W \rightarrow \mathbb{R}\), the dual map
      \(L^{\vee}\) associates the linear functional
    </p>
    <center>
      \(\displaystyle V \xrightarrow{L} W \xrightarrow{\alpha} \mathbb{R}.\)
    </center>
    <p>
      Thus, the dual map \(L^{\vee} : W^{\vee} \rightarrow V^{\vee}\) is given
      by
    </p>
    <center>
      \(\displaystyle L^{\vee} (\alpha) = \alpha \circ L \quad
      \operatorname{for} \alpha \in
W^{\vee} .\)
    </center>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">3.5</class>.
      </strong><i>(Functorial properties of the dual). Suppose \(V, W\) and
      \(S\) are real vector spaces.</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            If \(\mathbb{1}_V : V \rightarrow V\) is the identity map on
            \(V\), then \(\mathbb{1}_V^{\vee} : V^{\vee} \rightarrow
            V^{\vee}\) is the identity map on \(V^{\vee}\);
          </p>
        </li>
        <li>
          <p>
            If \(f : V \rightarrow W\) and \(g : W \rightarrow S\) are linear
            maps, then \((g \circ f)^{\vee} = f^{\vee} \circ g^{\vee}\).
          </p>
        </li>
      </ol></i>
    </div>
    <p>
      Fix a positive integer \(k\). For any linear map \(L : V \rightarrow W\)
      of vector spaces, define the pullback map \(L^{\ast} : A_k (W)
      \rightarrow A_k (V)\) to be 
    </p>
    <center>
      \(\displaystyle (L^{\ast} f) (v_1, \ldots, v_k) = f (L (v_1), \ldots, L
      (v_k))\)
    </center>
    <p>
      for \(f \in A_k (W)\) and \(v_1, \ldots, v_k \in V\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">3.6</class>. </strong><i>The
      pullback of covectors by a linear map satisfies the two functorial
      properties.</i>
    </p>
    <p>
      To each vector space \(V\), we associate the vector space \(A_k (V)\) of
      all \(k\)-covectors on \(V\), and to each linear map \(L : V \rightarrow
      W\) of vector spaces, we associate the pullback \(A_k (L) = L^{\ast} :
      A_k (W) \rightarrow A_k (V)\). Then \(A_k ()\) is a contravariant
      functor from the category of vector spaces and linear maps to itself.
    </p>
    <p>
      When \(k = 1\), for any vector space \(V\), the space \(A_1 (V)\) is the
      dual space, and for any linear map \(L : V \rightarrow W\), the pullback
      map \(A_1 (L) = L^{\ast}\) is the dual map \(L^{\vee} : W^{\vee}
      \rightarrow V^{\vee}\). Thus, the multicovector functor \(A_k ()\)
      generalizes the dual functor \(()^{\vee}\).
    </p>
    <p>
      
    </p>
    <p>
      Notes taken and exported by TeXmacs.
    </p>
  </body>
