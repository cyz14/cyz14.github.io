---
layout: post
title: Curvature and Differential Forms
date: 2026-06-27 16:50:00
description: chapter 2 of Differential Geometry, Connection, Curvature and Characteristic Classes
tags: connection
categories: math
tikzjax: true
---

<body>
    <h1 id="auto-1">2<span style="margin-left: 1em"></span>Curvature and Differential Forms<span style="margin-left: 1em"></span></h1>
    <p>
      第一章我们用向量场的语言发展了\(\mathbb{R}^3\)中曲线和曲面的曲率的经典理论。有一个对偶的使用微分形式的方法。微分形式即使我们只关心向量场也会自然出现。比如，开集上一个坐标系中切向量的系数是开集上的微分1形式。微分形式比向量场更微妙：它们可以被微分和相乘，并且它们在光滑映射的拉回下有意义。在1920s和1930s，&Eacute;lie
      Cartan引领了微分形式在微分几何中的应用，并且这些被证明是强大通用的工具。
    </p>
    <p>
      这一章中我们用微分形式重新发展联络和曲率的理论。首先，第10节把联络从一个切丛扩展到任意向量丛。11节我们展示如何用微分形式表示联络、曲率和挠率。最后，为了展示它们的用处，我们在12节用微分形式重新证明高斯绝妙定理。
    </p>
    <h2 id="auto-2">10<span style="margin-left: 1em"></span>Connections on a Vector Bundle<span style="margin-left: 1em"></span></h2>
    <p>
      仿射联络将方向导数从
      \(\mathbb{R}^n\)
      推广到一个任意流形，但是不包括沿着
      \(\mathbb{R}^n\)
      中一个子流形的向量场的方向导数的情形。为此，我们需要向量丛上联络的记号。一个流形上的仿射联络不过是切丛上的联络。因为向量丛上的联络
      \(\nabla_X s\)
      中两个变量的不对称性，挠率不再有定义，但是曲率仍然有意义。
    </p>
    <p>
      我们在向量丛上定义黎曼度量，使得流形上的黎曼度量变成切丛上的黎曼度量。联络和度量的相容性
      compactibility
      仍然有意义。但是，缺少挠率的概念让我们不能像黎曼流形上的黎曼联络那样再找出一个向量丛上唯一的联络。
    </p>
    <p>
      向量丛上的一个联络显然是一个局部算子，和其他所有局部算子一样，它可以被限制在任意开子集上。
    </p>
    <h3 id="auto-3">10.1<span style="margin-left: 1em"></span>Connections on a Vector Bundle<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">10.1</class>. </strong><i>Let \(E
      \rightarrow M\) be a \(C^{\infty}\) vector bundle over \(M\). A
      connection on \(E\) is a map</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \nabla : \mathfrak{X} (M) \times \Gamma (E)
        \rightarrow \Gamma (E)\)
      </center></i>
    </p>
    <p>
      <i>such that for \(X \in \mathfrak{X} (M)\) and \(s \in \Gamma
      (E)\),</i>
    </p>
    <i><ol>
      <li>
        <p>
          \(\nabla_X s\) is \(\mathcal{F}\)-linear in \(X\) and
          \(\mathbb{R}\)-linear in \(s\);
        </p>
      </li>
      <li>
        <p>
          (Leibniz rule) if \(f\) is a \(C^{\infty}\) function on \(M\), then
        </p>
        <center>
          \(\displaystyle \nabla_X (f s) = (X f) s + f \nabla_X s.\)
        </center>
      </li>
    </ol></i>
    <p>
      <i>Since \(X f = (d f) X\), the Leibniz rule may be rewritten as</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \nabla_X (f s) = (d f) (X) s + f \nabla_X s,\)
      </center></i>
    </p>
    <p>
      <i>or suppressing \(X\),</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \nabla (f s) = d f \cdot s + f \nabla s.\)
      </center></i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong> An affine connection on a manifold \(M\) is
      a connection on the tangent bundle \(T M \rightarrow M\).
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">10.2</class>. </strong>Let \(M\) be a
      submanifold of \(\mathbb{R}^n\) and \(E = T\mathbb{R}^n |_M\) the
      restriction of the tangent bundle of \(\mathbb{R}^n\) on \(M\). In
      Section 4.5, we defined the directional derivative
    </p>
    <center>
      \(\displaystyle D : \mathfrak{X} (M) \times \Gamma (T\mathbb{R}^n |_M)
      \rightarrow \Gamma
(T\mathbb{R}^n |_M) .\)
    </center>
    <p style="margin-bottom: 1em">
      By Proposition 4.9, it is a connection on the vector bundle
      \(T\mathbb{R}^n |_M\).
    </p>
    <p>
      We say that a section \(s \in \Gamma (E)\) is <b><i>flat</i></b> if
      \(\nabla_X s = 0\) for all \(X \in \mathfrak{X} (M)\).
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">10.3</class>. </strong>(Induced
      connection on a trivial bundle). Let \(E\) be a trivial bundle of rank
      \(r\) over a manifold \(M\). Thus, there is a bundle isomorphism \(\phi
      : E \rightarrow M \times \mathbb{R}^r\), called a
      <b><i>trivialization</i></b> for \(E\) over \(M\). The trivialization
      \(\phi : E \xrightarrow{\sim} M \times \mathbb{R}^r\) induces a
      connection on \(E\) as follows. If \(v_1, \ldots, v_r\) is a basis for
      \(\mathbb{R}^r\), then \(s_i : p \mapsto (p, v_i), i = 1, \ldots, r\),
      define a global frame for the product bundle \(M \times \mathbb{R}^r\)
      over \(M\), and \(e_i = \phi^{- 1} \circ s_i, i = 1, \ldots, r\), define
      a global frame for \(E\) over \(M\):
    </p>
    <p>
    <center>
    <script type="text/tikz" data-tex-packages='{ "amsmath": "", "amssymb": "", "amsfonts": "", "tikz-cd": "" }'>
    \begin{tikzcd}
    E                                     & M\times\mathbb{R}^r \arrow[l, "\phi^{-1}"'] \\
    M \arrow[u, "e_i"] \arrow[ru, "s_i"'] &                                            
    \end{tikzcd}
    </script>
    </center>
    </p>
    <p>
      So every section \(s \in \Gamma (E)\) can be written uniquely as a
      linear combination
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle s = \sum h^i e_i, \quad h^i \in \mathcal{F}.\)
      </center>
    </p>
    <p>
      We can define a connection \(\nabla\) on \(E\) by declaring the section
      \(e_i\) to be flat and applying the Leibniz rule and
      \(\mathbb{R}\)-linearity to define \(\nabla_X s\):
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \nabla_X s = \nabla_X \left( \sum
        h^i e_i \right) = \sum (X h^i) e_i .\)</td>
        <td align="right">(10.1)</td>
      </tr>
    </table>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">10.4</class>. </strong>Check that (10.1)
      defines a connection on the trivial bundle \(E\).
    </p>
    <p>
      The connection \(\nabla\) on a trivial bundle induced by a
      trivialization depends on the trivialization, for the flat sections for
      \(\nabla\) are precisely the sections of \(E\) corresponding to the
      constant sections of \(M \times \mathbb{R}^r\) under the trivialization.
    </p>
    <h3 id="auto-4">10.2<span style="margin-left: 1em"></span>Existence of a Connection on a Vector
    Bundle<span style="margin-left: 1em"></span></h3>
    <p>
      10.1
      节我们定义了向量丛上的联络，并展示了一个平凡丛上的联络。我们来证明一个任意向量丛上的联络的存在性。
    </p>
    <p>
      Let \(\nabla^0, \nabla^1\) be two connections on a vector bundle \(E\)
      over \(M\). By the Leibniz rule, for any vector field \(X \in
      \mathfrak{X} (M)\), section \(s \in \Gamma (E)\), and function \(f \in
      C^{\infty} (M)\),
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \nabla_X^0 (f s) = (X f) s + f
        \nabla_X^0 s,\)</td>
        <td align="right">(10.2)</td>
      </tr>
    </table>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \nabla_X^1 (f s) = (X f) s + f
        \nabla_X^1 s.\)</td>
        <td align="right">(10.3)</td>
      </tr>
    </table>
    <p>
      Hence,
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle (\nabla_X^0 + \nabla_X^1) (f s) = 2
        (X f) s + f (\nabla_X^0 + \nabla_X^1) s.\)</td>
        <td align="right">(10.4)</td>
      </tr>
    </table>
    <p>
      Because of the extra factor 2 in (10.4) the sum of two connections does
      not satisfy the Leibniz rule and so is not a connection. However, if we
      multiply (10.2) by \(1 - t\) and (10.3) by \(t\), then \((1 - t)
      \nabla_X^0 + t \nabla_X^1\) satisfies the Leibniz rule.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">10.5</class>. </strong><i>Any
      finite linear combination \(\sum t_i \nabla^i\) of connections
      \(\nabla^i\) is a connection provided the coefficients add up to 1,
      \(\sum t_i = 1\).</i>
    </p>
    <p>
      A finite linear combination whose coefficients add up to 1 is called a
      <b><i>convex linear combination</i></b>.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">10.6</class>. </strong><i>Every
      \(C^{\infty}\) vector bundle \(E\) over a manifold \(M\) has a
      connection.</i>
    </p>
    <h3 id="auto-5">10.3<span style="margin-left: 1em"></span>Curvature of a Connection on a Vector
    Bundle<span style="margin-left: 1em"></span></h3>
    <p>
      挠率的概念对任意一个向量丛上的联络不再有意义，但是曲率仍然有。它由仿射联络的相同的公式定义:
      for \(X, Y \in \mathfrak{X} (M)\) and \(s \in \Gamma (E)\),
    </p>
    <center>
      \(\displaystyle R (X, Y) s = \nabla_X \nabla_Y s - \nabla_Y \nabla_X s -
      \nabla_{[X, Y]} s \in
\Gamma (E) .\)
    </center>
    <p>
      So \(R\) is an \(\mathbb{R}\)-multilinear map
    </p>
    <center>
      \(\displaystyle \mathfrak{X} (M) \times \mathfrak{X} (M) \times \Gamma
      (E) \rightarrow \Gamma
(E) .\)
    </center>
    <p>
      As before, \(R (X, Y)\) is \(\mathcal{F}\)-linear in all three arguments
      and so it is actually defined pointwise. Moreover, because \(R_p (X_p,
      Y_p)\) is skew-symmetric in \(X_p\) and \(Y_p\), at every point \(p\)
      there is an alternating bilinear map
    </p>
    <center>
      \(\displaystyle R_p : T_p M \times T_p M \rightarrow \operatorname{Hom}
      (E_p, E_p) =:
\operatorname{End} (E_p)\)
    </center>
    <p>
      into the endomorphism ring of \(E_p\). We call this map the
      <b><i>curvature tensor</i></b> of the connection.
    </p>
    <h3 id="auto-6">10.4<span style="margin-left: 1em"></span>Riemannian Bundles<span style="margin-left: 1em"></span></h3>
    <p>
      We can also generalize the notion of a Riemannian metric to vector
      bundles. Let \(E \rightarrow M\) be a \(C^{\infty}\) vector bundle over
      a manifold \(M\). A <b><i>Riemannian metric</i></b> on \(E\) assigns to
      each \(p \in M\) an inner product \(\langle, \rangle_p\) on the fiber
      \(E_p\); the assignment is required to be \(C^{\infty}\) in the
      following sense: if \(s\) and \(t\) are \(C^{\infty}\) sections of
      \(E\), then \(\langle s, t \rangle\) is a \(C^{\infty}\) function on
      \(M\).
    </p>
    <p>
      Thus, a Riemannian metric on a manifold \(M\) is simply a Riemannian
      metric on the tangent bundle \(T M\). A vector bundle together with a
      Riemannian metric is called a <b><i>Riemannian bundle</i></b>.
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">10.7</class>. </strong>Let \(E\) be a
      trivial vector bundle of rank \(r\) over a maniold \(M\), with
      trivialization \(\phi : E \xrightarrow{\sim} M \times \mathbb{R}^r\).
      The Euclidean inner product \(\langle, \rangle_{\mathbb{R}^r}\) on
      \(\mathbb{R}^r\) induces a Riemannian metric on \(E\) via the
      trivialization \(\phi\): if \(u, v \in E_p\), then the fiber map
      \(\phi_p : E_p \rightarrow \mathbb{R}^r\) is a linear isomorphism and we
      define
    </p>
    <center>
      \(\displaystyle \langle u, v \rangle = \langle \phi_p (u), \phi_p (v)
      \rangle_{\mathbb{R}^r} .\)
    </center>
    <p style="margin-bottom: 1em">
      It is easy to check that \(\langle, \rangle\) is a Riemannian metric on
      \(E\).
    </p>
    <p>
      The proof of Theorem 1.12 generalizes to prove the existence of a
      Riemannian metric on a vector bundle.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">10.8</class>. </strong><i>On any
      \(C^{\infty}\) vector bundle \(\pi : E \rightarrow M\), there is a
      Riemannian metric.</i>
    </p>
    <h3 id="auto-7">10.5<span style="margin-left: 1em"></span>Metric Connections<span style="margin-left: 1em"></span></h3>
    <p>
      We say that a connection \(\nabla\) on a Riemannian bundle \(E\) is
      <b><i>compatible with the metric</i></b> if for all \(X \in \mathfrak{X}
      (M)\) and \(s, t \in \Gamma (E)\),
    </p>
    <center>
      \(\displaystyle X \langle s, t \rangle = \langle \nabla_X s, t \rangle +
      \langle s, \nabla_X t
\rangle .\)
    </center>
    <p>
      A connection compatible with the metric on a Riemannian bundle is also
      called a <b><i>metric connection</i></b>.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>Let \(E\) be a trivial vector bundle of rank
      \(r\) over a manifold \(M\), with trivialization \(\phi : E
      \xrightarrow{\sim} M \times \mathbb{R}^r\). We showed in Example 10.3
      that the trivializaiton induces a connection \(\nabla\) on \(E\) and in
      Example 10.7 that the trivialization induces a Riemannian metric on
      \(E\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">10.9</class>. </strong><i>On a
      trivial vector bundle \(E\) over a manifold \(M\) with trivialization
      \(\phi : E \xrightarrow{\sim} M \times \mathbb{R}^r\), the connection
      \(\nabla\) on \(E\) induced by the trivialization \(\phi\) is compatible
      with the Riemannian metric \(\langle, \rangle\) on \(E\) induced by the
      trivialization.</i>
    </p>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">10.10</class>. </strong>(Connection on
      \(T\mathbb{R}^n |_M\)). If \(M\) is a submanifold of \(\mathbb{R}^n\),
      the Euclidean metric on \(\mathbb{R}^n\) restricts to a Riemannian
      metric on the vector bundle \(T\mathbb{R}^n |_M\). Sections of the
      vector bundle \(T\mathbb{R}^n |_M\) are vector fields along \(M\) in
      \(\mathbb{R}^n\). As noted in Example 10.2, the directional derivative
      in \(\mathbb{R}^n\) induces a connection
    </p>
    <center>
      \(\displaystyle D : \mathfrak{X} (M) \times \Gamma (T\mathbb{R}^n |_M)
      \rightarrow \Gamma
(T\mathbb{R}^n |_M) .\)
    </center>
    <p style="margin-bottom: 1em">
      Proposition 4.10 asserts that the conneciton \(D\) on \(T\mathbb{R}^n
      |_M\) has zero curvature and is compatible with the metric.
    </p>
    <p>
      The Gauss curvature equation for a surface \(M\) in \(\mathbb{R}^3\), a
      key ingredient of the proof of Gauss's Theorema Egregium, is a
      consequence of the vanishing of the curvature tensor of the conneciton
      \(D\) on the bundle \(T\mathbb{R}^3 |_M\) (Theorem 8.1).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>引理 <class style="font-style: normal">10.11</class>. </strong><i>Let
      \(E \rightarrow M\) be a Riemannian bundle. Suppose \(\nabla^1, \ldots,
      \nabla^m\) are connections on \(E\) compatible with the metric and
      \(a_1, \ldots, a_m\) are \(C^{\infty}\) functions on \(M\) that add up
      to 1. Then \(\nabla = \sum_i a_i \nabla^i\) is a connection on \(E\)
      compatible with the metric.</i>
    </p>
    <p style="margin-top: 1em">
      <strong>证明. </strong>By Proposition 10.5, \(\nabla\) is a
      connection on \(E\). It remains to check that \(\nabla\) is compatible
      with the metirc. If \(X \in \mathfrak{X} (M)\) and \(s, t \in \Gamma
      (E)\), then
    </p>
    <center>
      \(\displaystyle X \langle s, t \rangle = \langle \nabla_X^i s, t \rangle
      + \langle s,
\nabla_X^i t \rangle\)
    </center>
    <p>
      for all \(i\) because \(\nabla^i\) is compatible with the metric. Now
      multiply (10.5) by \(a_i\) and sum
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  X \langle s, t \rangle & = \sum a_i
      X \langle s, t \rangle\\
  & = \left\langle \sum a_i \nabla_X^i s, t
      \right\rangle + \left\langle s,
  \sum a_i \nabla_X^i t \right\rangle\\

      & = \langle \nabla_X s, t \rangle + \langle s, \nabla_X t \rangle
      .

\end{array}\)
</center>
<p style="margin-bottom: 1em">
<span style="margin-left: 1em"></span>\(\Box\)
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">10.12</class>. </strong><i>On any
Riemannian bundle \(E \rightarrow M\), there is a connection compatible
with the metric.</i>
</p>
<h3 id="auto-8">10.6<span style="margin-left: 1em"></span>Restricting a Connection to an Open
Subset<span style="margin-left: 1em"></span></h3>
<p>
A connection \(\nabla\) on a vector bundle \(E\) over \(M\)
</p>
<center>
\(\displaystyle \nabla : \mathfrak{X} (M) \times \Gamma (E) \rightarrow
\Gamma (E)\)
</center>
<p>
is \(\mathcal{F}\)-linear in the first argument, but not
\(\mathcal{F}\)-linear in the second argument. However, it turns out
that the \(\mathcal{F}\)-linearity in the first argument and the Leibniz
rule in the second argument are enough to imply that \(\nabla\) is a
local operator.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">10.13</class>. </strong><i>Let
\(\nabla\) be a connection on a vector bundle \(E\) over a manifold
\(M\), \(X\) be a smooth vector field on \(M\), and \(s\) a smooth
section of \(E\). If either \(X\) or \(s\) vanishes identically on an
open subset \(U\), then \(\nabla*X s\) vanishes identically on
\(U\).</i>
</p>
<p>
A conneciton on a vector bundle can be restricted to any open subset.
Given a connection \(\nabla\) on a vector bundle \(E\), for every open
set \(U\) there is a connection
</p>
<center>
\(\displaystyle \nabla^U : \mathfrak{X} (U) \times \Gamma (U, E)
\rightarrow \Gamma (U, E)\)
</center>
<p>
such that for any global vector field \(\bar{X} \in \mathfrak{X} (M)\)
and global section \(\bar{s} \in \Gamma (E)\),
</p>
<center>
\(\displaystyle \nabla*{\bar{X} |_U}^U (\bar{s} |\_U) = (\nabla_{\bar{X}}
\bar{s}) |_U .\)
</center>
<p>
Suppose \(X \in \mathfrak{X} (U)\) and \(s \in \Gamma (U, E)\). For any
\(p \in U\), to define \(\nabla_X^U s \in \Gamma (U, E)\) first pick a
global vector field \(\bar{X}\) and a global section \(\bar{s} \in
\Gamma (E)\) that agrees with \(X\) and \(s\) in a neighborhood of
\(p\). Then define
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle (\nabla_X^U s) (p) =
(\nabla_{\bar{X}} \bar{s}) (p) .\)</td>
<td align="right">(10.5)</td>
</tr>
</table>
<p>
Because \(\nabla*{\bar{X}} \bar{s}\) is a local operator in \(\bar{X}\)
and in \(\bar{s}\), this definition is independent of the choice of
\(\bar{X}\) and \(\bar{s}\). It is a routing matter to show that
\(\nabla^U\) satisfies all the properties of a connection on \(E |\_U\).
</p>
<h3 id="auto-9">10.7<span style="margin-left: 1em"></span>Connections at a Point<span style="margin-left: 1em"></span></h3>
<p>
Suppose \(\nabla\) is a connection on a vector bundle \(E\) over a
manifold \(M\). For \(X \in \mathfrak{X} (M)\) and \(s \in \Gamma (E)\),
since \(\nabla_X s\) is \(\mathcal{F}\)-linear in \(X\), it is a point
operator in \(X\) and Proposition 7.25 assures us that it can be defined
pointwise in \(X\): there is a unique map, also denoted by \(\nabla\),
</p>
<center>
\(\displaystyle \nabla : T_p M \times \Gamma (E) \rightarrow E_p\)
</center>
<p>
such that if \(X \in \mathfrak{X} (M)\) and \(s \in \Gamma (E)\), then
</p>
<center>
\(\displaystyle \nabla*{X*p} s = (\nabla_X s)\_p .\)
</center>
<p>
It is easy to check that \(\nabla*{X*p} s\) has the following
properties: for \(X_p \in T_p M\) and \(s \in \Gamma (E)\),
</p>
<ol>
<li>
<p>
\(\nabla*{X*p} s\) is \(\mathbb{R}\)-linear in \(X_p\) and in \(s\);
</p>
</li>
<li>
<p>
if \(f\) is a \(C^{\infty}\) function on \(M\), then
</p>
<center>
\(\displaystyle \nabla*{X*p} (f s) = (X_p f) s (p) + f (p)
\nabla*{X*p} s.\)
</center>
</li>
</ol>
<h2 id="auto-10">11<span style="margin-left: 1em"></span>Connection, Curvature, and Torsion
Forms<span style="margin-left: 1em"></span></h2>
<p>
根据高斯绝妙定理，如果
\(R_p\) 是 \(\mathbb{R}^3\) 中曲面 \(M\)
点 \(p\)
处的曲率张量，\(u, v\)
是切平面 \(T_p M\)
的任意正交标架，那么这点的高斯曲率是
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle K_p = \langle R_p (u, v) v, u
\rangle .\)</td>
<td align="right">(11.1)</td>
</tr>
</table>
<p>
因为这个公式不依赖于曲面在
\(\mathbb{R}^3\)
中的嵌入，而仅依赖曲面的黎曼结构，它对任意2维黎曼流形有意义，可以作为这样一个曲面在一点处高斯曲率的定义，比如，双曲上半平面
\(\mathbb{H}^2\)。为了用 (11.1)
计算高斯曲率，我们需要先用含六项的公式
(6.8) 计算黎曼联络
\(\nabla\)，然后计算曲率张量
\(R_p (u, v)
v\)，显然是一个非平凡的任务。
</p>
<p>
微分形式的一个重要优势就是它的可计算性，因此这一节我们用微分形式重新介绍联络、曲率和挠率。这会给下一节双曲上半平面的曲率带来简单的计算。
</p>
<p>
相对一个向量丛的标架，丛上的联络可以表示为1形式的矩阵，曲率为2形式的矩阵。这两个矩阵间的关系是联络的结构方程。
</p>
<p>
线性代数中的 Gram-Schmidt
过程把一个黎曼丛的任意
\(C^{\infty}\)
标架转为一个正交标架。相对这个正交标架，度量联络的联络矩阵是反对称的。根据结构方程，曲率矩阵也是反对称的。度量联络的曲率矩阵的反对称性在后面章节会有重要结果。
</p>
<p>
在切丛上，联络的挠率可以考试为2形式的矩阵，叫作挠率形式。有一个结构方程联系挠率形式和对偶形式以及联络形式。
</p>
<h3 id="auto-11">11.1<span style="margin-left: 1em"></span>Connection and Curvature Forms<span style="margin-left: 1em"></span></h3>
<p>
Let \(\nabla\) be a connection on a \(C^{\infty}\) rank \(r\) vector
bundle \(\pi : E \rightarrow M\). We are interested in describing
\(\nabla\) locally. Section 10.6 shows how on every open subset \(U\) of
\(M\), \(\nabla\) restricts to a connection on \(E |\_U \rightarrow U\):
</p>
<center>
\(\displaystyle \nabla^U : \mathfrak{X} (U) \times \Gamma (U, E)
\rightarrow \Gamma (U, E) .\)
</center>
<p>
We will usually omit the superscript \(U\) and write \(\nabla^U\) as
\(\nabla\).
</p>
<p>
Suppose \(U\) is a trivializing open set for \(E\) and \(e_1, \ldots,
e_r\) is a frame for \(E\) over \(U\) (Proposition 7.22), and let \(X
\in \mathfrak{X} (M)\) be a \(C^{\infty}\) vector field on \(U\). On
\(U\), since any section \(s \in \Gamma (U, E)\) is a linear combination
\(s = \sum a^j e_j\), the section \(\nabla_X s\) can be computed from
\(\nabla_X e_j\) by linearity and the Leibniz rule. As a section of
\(E\) over \(U\), \(\nabla_X e_j\) is a linear combination of the
\(e_i\)'s with coefficients \(\omega_j^i\) depending on \(X\):
</p>
<center>
\(\displaystyle \nabla_X e_j = \sum \omega_j^i (X) e_i .\)
</center>
<p>
The \(\mathcal{F}\)-linearity of \(\nabla_X e_j\) in \(X\) implies that
\(\omega_j^i\) is \(\mathcal{F}\)-linear in \(X\) and so \(\omega_j^i\)
is a 1-form on \(U\) (Corollary 7.27). The 1-forms \(\omega_j^i\) on
\(U\) are called the <b><i>connection forms</i></b>, and the matrix
\(\omega = [\omega_j^i]\) is called the <b><i>connection matrix</i></b>,
of the connection \(\nabla\) relative to the frame \(e_1, \ldots, e_r\)
on \(U\).
</p>
<p>
Similarly, for \(X, Y \in \mathfrak{X} (M)\), the section \(R (X, Y)
e_j\) is a linear combination of \(e_1, \ldots, e_r\):
</p>
<center>
\(\displaystyle R (X, Y) e_j = \sum \Omega_j^i (X, Y) e_i .\)
</center>
<p>
Since
</p>
<center>
\(\displaystyle R (X, Y) = \nabla_X \nabla_Y - \nabla_Y \nabla_X -
\nabla*{[X, Y]}\)
</center>
<p>
is alternating and is \(\mathcal{F}\)-bilinear, so is \(\Omega*j^i\). By
Section 7.8, \(\Omega_j^i\) is a 2-form on \(U\). The 2-forms
\(\Omega_j^i\) are called the <b><i>curvature forms</i></b>, and the
matrix \(\Omega = [\Omega_j^i]\) is called the <b><i>curvature
matrix</i></b>, of the connection \(\nabla\) relative to the frame
\(e_1, \ldots, e_r\) on \(U\).
</p>
<p>
Recall that is \(\alpha, \beta\) are \(C^{\infty}\) 1-forms and \(X, Y\)
are \(C^{\infty}\) vector fields on a manifold, then
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle (\alpha \wedge \beta) (X, Y) =
\alpha (X) \beta (Y) - \alpha (Y) \beta (X)\)</td>
<td align="right">(11.2)</td>
</tr>
</table>
<p>
and
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle (d \alpha) (X, Y) = X \alpha (Y) - Y
\alpha (X) - \alpha ([X, Y])\)</td>
<td align="right">(11.3)</td>
</tr>
</table>
<p style="margin-top: 1em">
<strong>定理 <class style="font-style: normal">11.1</class>. </strong><i>Let
\(\nabla\) be a connection on a vector bundle \(E \rightarrow M\) of
rank \(r\). Relative to a frame \(e_1, \ldots, e_r\) for \(E\) over a
trivializing open set \(U\), the curvature form \(\Omega_j^i\) are
related to the connection forms \(\omega_j^i\) by the <b><i>second
structural equation</i></b>:</i>
</p>
<p style="margin-bottom: 1em">
<i><center>
\(\displaystyle \Omega_j^i = d \omega_j^i + \sum_k \omega_k^i \wedge
\omega_j^k .\)
</center></i>
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>注意 <class style="font-style: normal">11.2</class>. </strong>(The
Einstein summation convention). Einstein summation convention.
</p>
<p style="margin-top: 1em">
<strong>证明. </strong>(Theorem 11.1). Let \(X\) and \(Y\)
be smooth vector fields on \(U\). Then
</p>
<center>
\(\displaystyle \begin{array}{rl}
\nabla_X \nabla_Y e_j & = \nabla_X
\sum_k (\omega_j^k (Y) e_k)\\
& = \sum_k X \omega_j^k (Y) e_k + \sum_k
\omega_j^k (Y) \nabla_X e_k\\
& = \sum_i X \omega_j^i (Y) e_i +
\sum*{i, k} \omega*j^k (Y) \omega_k^i (X)
e_i .
\end{array}\)
</center>
<p>
Interchanging \(X, Y\) gives
</p>
<center>
\(\displaystyle \nabla_Y \nabla_X e_j = \sum_i Y \omega_j^i (X) e_i +
\sum*{i, k} \omega*j^k
(X) \omega_k^i (Y) e_i .\)
</center>
<p>
Furthermore,
</p>
<center>
\(\displaystyle \nabla*{[X, Y]} e*j = \sum_i \omega_j^i ([X, Y]) e_i .\)
</center>
<p>
Hence, in Einstein notation,
</p>
<center>
\(\displaystyle \begin{array}{rl}
R (X, Y) e_j = & \nabla_X \nabla_Y
e_j - \nabla_Y \nabla_X e_j - \nabla*{[X,
Y]} e_j\\
= & (X
\omega_j^i (Y) - Y \omega_j^i (X) - \omega_j^i ([X, Y])) e_i\\
& +
(\omega_k^i (X) \omega_j^k (Y) - \omega_k^i (Y) \omega_j^k (X)) e_i\\

      = & d \omega_j^i (X, Y) e_i + \omega_k^i \wedge \omega_j^k (X, Y) e_i
      \qquad

(\operatorname{by} (11.3) \operatorname{and} (11.2))\\
= & (d
\omega*j^i + \omega_k^i \wedge \omega_j^k) (X, Y) e_i .
\end{array}\)
</center>
<p>
Comparing this with the definition of the curvature form \(\Omega_j^i\)
gives
</p>
<p style="margin-bottom: 1em">
<center>
\(\displaystyle \Omega_j^i = d \omega_j^i + \sum_k \omega_k^i \wedge
\omega_j^k .\)
</center>
<span style="margin-left: 1em"></span>
\(\Box\)
</p>
<h3 id="auto-12">11.2<span style="margin-left: 1em"></span>Connections on a Framed Open Set<span
    style="margin-left: 1em"></span></h3>
<p>
Suppose \(E\) is a \(C^{\infty}\) vector bundle over a manifold \(M\)
and \(U\) is an open set on which there is a \(C^{\infty}\) frame \(e_1,
\ldots, e_r\) for \(E\). We call \(U\) a <b><i>framed open set</i></b>
for \(E\) for short. A connection \(\nabla\) on \(E |\_U\) determines a
unique connection matrix \([\omega_j^i]\) relative to the frame \(e_1,
\ldots, e_r\). Conversely, any matrix of 1-forms \([\omega_j^i]\) on
\(U\) determines a connection on \(E |\_U\) as follows.
</p>
<p>
Given a matrix \([\omega_j^i]\) of 1-forms on \(U\), and \(X, Y \in
\mathfrak{X} (U)\), we set
</p>
<center>
\(\displaystyle \nabla_X e_j = \sum \omega_j^i (X) e_i,\)
</center>
<p>
and define \(\nabla_X Y\) by applying the Leibniz rule to \(Y = \sum h^j
e_j\):
</p>
<center>
\(\displaystyle \begin{array}{rl}
\nabla_X Y & = \nabla_X (h^j e_j) =
(X h^j) e_j + h^j \omega_j^i (X) e_i\\
& = ((X h^i) + h^j \omega_j^i
(X)) e_i . \hspace{3cm} \text{(11.4)}
\end{array}\)
</center>
<p>
With this definition, \(\nabla\) is a connection on \(E |\_U\).
</p>
<h3 id="auto-13">11.3<span style="margin-left: 1em"></span>The Gram-Schmidt Process<span style="margin-left: 1em"></span></h3>
<p>
线性代数里的 Gram-Schmidt
过程把内积空间 \(V\)
中的任意线性无关向量
\(v_1, \ldots, v_n\)
转为一个展成相同空间的正交向量集合。
\(\operatorname{proj}\_a b\) 表示把 \(b\)
正交投影到 \(a\)
的线性空间。那么,
</p>
<center>
\(\displaystyle \operatorname{proj}\_a b = \frac{\langle b, a
\rangle}{\langle a, a \rangle} a.\)
</center>
<p>
To carry out the Gram-Schimidt process, we first create an orthonormal
set \(w_1, \ldots, w_n\):
</p>
<center>
\(\displaystyle \begin{array}{rl}
w_1 & = v_1,\\
w_2 & = v_2
-\operatorname{proj}*{w*1} v_2,\\
& = v_2 - \frac{\langle v_2, w_1
\rangle}{\langle w_1, w_1 \rangle} w_1,\\
w_3 & = v_3
-\operatorname{proj}*{w*1} v_3 -\operatorname{proj}*{w_2} v_3

      \hspace{3cm}   \text{(11.5)}\\

& = v_3 - \frac{\langle v_3, w_1
\rangle}{\langle w_1, w_1 \rangle} w_1 -
\frac{\langle v_3, w_2
\rangle}{\langle w_2, w_2 \rangle} w_2,
\end{array}\)
</center>
<p>
and so on.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">11.3</class>. </strong><i>In the
Gram-Schmidt process, for each \(k\), the set \(w_1, \ldots, w_k\) span
the same linear subspace of \(V\) as \(v_1, \ldots, v_k\).</i>
</p>
<p>
To get an orthonormal set \(e_1, \ldots, e_n\), define
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle e_i = \frac{w_i}{\| w_i \|} .\)</td>
<td align="right">(11.6)</td>
</tr>
</table>
<p>

    </p>
    <h3 id="auto-14">11.4<span style="margin-left: 1em"></span>Metric Connection Relative to an
    Orthonormal Frame<span style="margin-left: 1em"></span></h3>
    <p>
      前面小节我们看到向量丛
      \(E \rightarrow M\)
      的有标架的开集 \(U\)
      上，一个联络完全由
      \(U\) 上的1形式矩阵 \([\omega_j^i]\)
      确定。假设向量丛 \(E
      \rightarrow M\)
      带有一个黎曼度量。度量联络的性质可以被翻译为正交标架上联络矩阵
      \([\omega_j^i]\) 的条件。
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">11.4</class>. </strong><i>Let \(E
      \rightarrow M\) be a Riemannian bundle and \(\nabla\) a connection on
      \(E\).</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            If the connection \(\nabla\) is compatible with the metric, then
            its connection matrix \([\omega_j^i]\) relative to any orthonormal
            frame \(e_1, \ldots, e_r\) for \(E\) over a trivializing open set
            \(U \subset M\) is skew-symmetric.
          </p>
        </li>
        <li>
          <p>
            If every point \(p \in M\) has a trivializing neighborhood \(U\)
            for \(E\) such that the connection matrix \([\omega_j^i]\)
            relative to an orthonormal frame \(e_1, \ldots, e_r\) for \(E\)
            over \(U\) is skew-symmetric, then the connection \(\nabla\) is
            compatible with the metric.
          </p>
        </li>
      </ol></i>
    </div>
    <p style="margin-top: 1em">
      <strong>证明. </strong>
    </p>
    <ol>
      <li>
        <p>
          Suppose \(\nabla\) is compatible with the metric. For all \(X \in
          \mathfrak{X} (M)\) and \(i, j\),
        </p>
        <center>
          \(\displaystyle \begin{array}{rl}

0 = X \langle e*i, e_j \rangle &
= \langle \nabla_X e_i, e_j \rangle +
\langle e_i, \nabla_X e_j
\rangle\\
& = \langle \omega_i^k (X) e_k, e_j \rangle + \langle
e_i, \omega_j^k (X)
e_k \rangle\\
& = \omega_i^k (X) \delta*{k
j} + \omega_j^k (X) \delta i k\\
& = \omega_i^j (X) + \omega_j^i
(X) .
\end{array}\)
</center>
<p>
Hence,
</p>
<center>
\(\displaystyle \omega_i^j = - \omega_j^i .\)
</center>
</li>
<li>
<p>
We note first that compatibility with the metric is a local
condition, so \(\nabla\) is compatible with the metric if and only
if its restriction \(\nabla^U\) to any open set \(U\) is compatible
with the metric. Suppose \(\omega_i^j = - \omega_j^i\). Let \(s =
\sum a^i e_i\) and \(t = \sum b^j e_j\), with \(a^i, b^j \in
C^{\infty} (U)\). Then
</p>
<center>
\(\displaystyle \begin{array}{rl}
X \langle s, t \rangle & = X
\left( \sum a^i b^i \right) = \sum (X a^i) b^i

- \sum a^i X b^i
  .\\
  \nabla_X s & = \nabla_X (a^i e_i) = (X a^i) e_i + a^i \nabla_X
  e_i\\
  & = (X a^i) e_i + a^i \omega_i^k (X) e_k,\\
  \langle
  \nabla_X s, t \rangle & = \sum (X a^i) b^i + \sum a^i \omega_i^j
  (X)
  b^j \hspace{3cm} \text{(11.7)}\\
  \langle s, \nabla_X t
  \rangle & = \sum (X b^i) a^i + \sum b^i \omega_i^j (X)
  a^j\\
  & =
  \sum (X b^i) a^i + \sum a^i b^j \omega_j^i (X) . \hspace{3cm}

            \text{(11.8)}

  \end{array}\)
  </center>
  </li>
  </ol>
  <p>
  But by the skew-symmetric of \(\omega\),
  </p>
  <center>
  \(\displaystyle \sum a^i b^j \omega_i^j (X) + \sum a^i b^j \omega_j^i
  (X) = 0.\)
  </center>
  <p>
  Hence, adding (11.7) and (11.8) gives
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  \langle \nabla_X s, t \rangle +
  \langle s, \nabla_X t \rangle & = \sum (X
  a^i) b^i + \sum (X b^i)
  a^i\\
  & = X \langle s, t \rangle .
  \end{array}\)
  </center>
  <p style="margin-bottom: 1em">
  <span style="margin-left: 1em"></span>\(\Box\)
  </p>
  <p style="margin-top: 1em; margin-bottom: 1em">
  <strong>命题 <class style="font-style: normal">11.5</class>. </strong><i>If the
  connection matrix \([\omega_j^i]\) relative to a frame \(e_1, \ldots,
  e_n\) of an affine connection on a manifold is skew-symmetric, then so
  is the curvature matrix \([\Omega_j^i]\).</i>
  </p>
  <p style="margin-top: 1em">
  <strong>证明. </strong>By the structural equation,
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  \Omega_i^j & = d \omega_i^j +
  \omega_k^j \wedge \omega_i^k\\
  & = - d \omega_j^i + (- \omega_j^k)
  \wedge (- \omega_k^i)\\
  & = - d \omega_j^i - \omega_k^i \wedge
  \omega_j^k\\
  & = - \Omega_j^i .
  \end{array}\)
  </center>
  <p style="margin-bottom: 1em">
  <span style="margin-left: 1em"></span>\(\Box\)
  </p>
  <h3 id="auto-15">11.5<span style="margin-left: 1em"></span>Connections on the Tangent Bundle<span
      style="margin-left: 1em"></span></h3>
  <p>
  A connection on the tangent bundle \(T M\) of a manifold \(M\) is simply
  an affine connection on \(M\). In addition to the curvature tensor \(R
  (X, Y ) Z\), an affine connection has a torsion tensor \(T (X, Y)\).
  </p>
  <p>
  Let \(U\) be an open set in \(M\) on which the tangent bundle \(T M\)
  has a smooth frame \(e_1, \ldots, e_n\). If \(U\) is a coordinate open
  set with coordinates \(x^1, \ldots, x^n\), then \(\partial / \partial
  x^1, \ldots, \partial / \partial x^n\) is such a frame, but we will
  consider the more general setting where \(U\) need not be a coordinate
  open set. Let \(\theta^1, \ldots, \theta^n\) be the dual frame of
  1-forms on \(U\); this means
  </p>
  <center>
  \(\displaystyle \theta^i (e_j) = \delta_j^i .\)
  </center>
  <p style="margin-top: 1em; margin-bottom: 1em">
  <strong>命题 <class style="font-style: normal">11.6</class>. </strong><i>If
  \(X\) is a smooth vector field on the open set \(U\), then \(X = \sum
  \theta^i (X) e_i\).</i>
  </p>
  <p>
  For \(X, Y \in \mathfrak{X} (U)\), the torsion \(T (X, Y)\) is a linear
  combination of the vector fields \(e_1, \ldots, e_n\), so we can write
  </p>
  <center>
  \(\displaystyle T (X, Y) = \sum \tau^i (X, Y) e_i .\)
  </center>
  <p>
  Since \(T (X, Y)\) is alternating and \(\mathcal{F}\)-bilinear, so are
  the coefficients \(\tau^i\). Therefore, the \({\tau^i}' s\) are 2-forms
  on \(U\), called the <b><i>torsion forms</i></b> on the affine
  connection \(\nabla\) relative to the frame \(e_1, \ldots, e_n\) on
  \(U\).
  </p>
  <p>
  Since the torsion and curvature forms are determined completely by the
  frame \(e_1, \ldots, e_n\) and the connection, there should be formulas
  for \(\tau^i\) and \(\Omega_j^i\) in terms of the dual forms and the
  connection forms. Indeed, Theorem 11.1 expresses the curvature forms in
  terms of the connection forms alone.
  </p>
  <p style="margin-top: 1em">
  <strong>定理 <class style="font-style: normal">11.7</class>.
  </strong><i>(Structural equations).Relative to frame \(e_1, \ldots,
  e_n\) for the tangent bundle over an open set \(U\) of a manifold \(M\),
  the torsion and curvature form of an affine connection on \(M\) can be
  given in terms of the dual 1-forms and the connection forms:</i>
  </p>
  <div style="margin-bottom: 1em">
  <i><ol>
  <li>
  <p>
  (the first structural equation) \(\tau^i = d \theta^i + \sum_j
  \omega_j^i \wedge \theta^j\);
  </p>
  </li>
  <li>
  <p>
  (the second structural equation) \(\Omega_j^i = d \omega_j^i +
  \sum_k \omega_k^i \wedge \omega_j^k\).
  </p>
  </li>
  </ol></i>
  </div>
  <p style="margin-top: 1em">
  <strong>证明. </strong>The second structural equation is a
  special case of Theorem 11.1, which is true more generally for any
  vector bundle. The proof of the first structural equation is a matter of
  unraveling the definition of the torsion. Let \(X, Y\) be smooth vector
  fields on \(U\). By Proposition 11.6, we can write
  </p>
  <center>
  \(\displaystyle Y = \sum \theta^j (Y) e_j .\)
  </center>
  <p>
  Then
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  \nabla_X Y & = \nabla_X (\theta^j
  (Y) e_j)\\
  & = (X \theta^j (Y)) e_j + \theta^j (Y) \nabla_X e_j\\
  &
  = (X \theta^j (Y)) e_j + \theta^j (Y) \omega_j^i (X) e_i
  \end{array}\)
  </center>
  <p>
  By symmetry,
  </p>
  <center>
  \(\displaystyle \nabla_Y X = (Y \theta^j (X)) e_j + \theta^j (X)
  \omega_j^i (Y) e_i .\)
  </center>
  <p>
  Finally, by Proposition 11.6 again,
  </p>
  <center>
  \(\displaystyle [X, Y] = \theta^i ([X, Y]) e_i .\)
  </center>
  <p>
  Thus,
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  T (X, Y) & = \nabla_X Y - \nabla_Y X - [X, Y]\\
  & = ((X \theta^i (Y) - Y \theta^i (X) - \theta^i ([X, Y]) ) + (\omega_j^i
  (X) \theta^j (Y) - \omega_j^i (Y) \theta^j (X)) ) e_i\\

        & = (d \theta^i + \omega_j^i \wedge \theta^j) (X, Y) e_i,

  \end{array}\)
  </center>
  <p>
  where the last equality follows from (11.3) and (11.2). Hence,
  </p>
  <p style="margin-bottom: 1em">
  <center>
  \(\displaystyle \tau^i = d \theta^i + \sum \omega_j^i \wedge \theta^j
  .\)
  </center>
  <span style="margin-left: 1em"></span>
  \(\Box\)
  </p>
  <p>
  我们现在可以把黎曼联络的定义的两个性质转为矩阵
  \([\omega_j^i]\) 的条件。
  </p>
  <p style="margin-top: 1em">
  <strong>命题 <class style="font-style: normal">11.8</class>. </strong><i>Let
  \(M\) be an Riemannian manifold and \(U\) an open subset on which there
  is an orthonormal frame \(e_1, \ldots, e_n\). Let \(\theta^1, \ldots,
  \theta^n\) be the dual frame of 1-forms. Then there exists a unique
  skew-symmetric matrix \([\omega_j^i]\) of 1-forms such that</i>
  </p>
  <div style="margin-bottom: 1em">
  <i><table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle d \theta^i + \sum_j \omega_j^i
  \wedge \theta^j = 0 \quad
  \operatorname{for}\operatorname{all}i = 1,
  \ldots, n.\)</td>
  <td align="right">(11.9)</td>
  </tr>
  </table></i>
  </div>
  <p style="margin-top: 1em">
  <strong>证明. </strong>In Theorem 6.6 we showed the
  existence of a Riemannian connection \(\nabla\) on any manifold. Let
  \([\omega_j^i]\) be the connection matrix of \(\nabla\) relative to the
  orthonormal frame \(e_1, \ldots, e_n\) on \(U\). Because \(\nabla\) is
  compatible with the metric, by Proposition 11.4, the matrix
  \([\omega_j^i]\) is skew-symmetric. Because \(\nabla\) is torsion-free,
  by Theorem 11.7, it satisfies
  </p>
  <center>
  \(\displaystyle d \theta^i + \sum_j \omega_j^i \wedge \theta^j = 0.\)
  </center>
  <p>
  This proves the existence of the matrix \([\omega_j^i]\) with the two
  required properties.
  </p>
  <p style="margin-bottom: 1em">
  To prove uniqueness, suppose \([\omega_j^i]\) is any skew-symmetric
  matrix of 1-forms on \(U\) satisfying (11.9). In Section 11.2, taking
  the vector bundle \(E\) to be the tangent bundle \(T M\), we showed that
  \([\omega_j^i]\) defines an affine connection \(\nabla\) on \(U\) of
  which it is the connection matrix relative to the frame \(e_1, \ldots,
  e_n\). Because \([\omega_j^i]\) is skew-symmetric, \(\nabla\) is
  compatible with the metric (Proposition 11.4), and because
  \([\omega_j^i]\) satisfies the equation (11.9), \(\nabla\) is
  torsion-free (Theorem 11.7). Thus, \(\nabla\) is the unique Riemannian
  connection on \(U\).<span style="margin-left: 1em"></span>\(\Box\)
  </p>
  <p>
  If \(A = [\alpha_j^i]\) and \(B = [\beta_j^i]\) are matrices of
  differential forms on \(M\) with the number of columns of \(A\) equal to
  the number of rows of \(B\), then their wedge product \(A \wedge B\) is
  defined to be the matrix of differential forms whose \((i, j)\)-entry is
  </p>
  <center>
  \(\displaystyle (A \wedge B)^i_j = \sum_k \alpha_k^i \wedge \beta_j^k,\)
  </center>
  <p>
  and \(d A\) is defined to be \([d \alpha_j^i]\). In matrix notation, we
  write
  </p>
  <center>
  \(\displaystyle \begin{array}{llll}
  \tau = \left[ \begin{array}{c}

        \tau^1 \\
      \vdots\\
      \tau^n

  \end{array} \right], & \theta =
  \left[ \begin{array}{c}
  \theta^1\\
  \vdots\\
  \theta^n

        \end{array} \right], & \omega = [\omega_j^i], & \Omega = [\Omega_j^i]
        .

  \end{array}\)
  </center>
  <p>
  Then the first structural equation becomes
  </p>
  <center>
  \(\displaystyle \tau = d \theta + \omega \wedge \theta\)
  </center>
  <p>
  and the second structural equation becomes
  </p>
  <center>
  \(\displaystyle \Omega = d \omega + \omega \wedge \omega .\)
  </center>
  <p style="margin-top: 1em">
  <strong>例. </strong>Connection and curvature forms on the
  Poincare disk
  </p>
  <center>
  \(\displaystyle \mathbb{D}= \{ z = x + i y \in \mathbb{C}| | z | < 1
  \}\)
  </center>
  <p>
  in the complex plane with Riemannian metric
  </p>
  <center>
  \(\displaystyle \langle, \rangle*z = \frac{4 (d x \otimes d x + d y
  \otimes d y)}{(1 - | z
  |^2)^2} = \frac{4 (d x \otimes d x + d y \otimes
  d y)}{(1 - x^2 - y^2)^2} .\)
  </center>
  <p>
  An orthonormal frame for \(\mathbb{D}\) is
  </p>
  <center>
  \(\displaystyle e_1 = \frac{1}{2} (1 - | z |^2) \frac{\partial}{\partial
  x}, \quad e_2 =
  \frac{1}{2} (1 - | z |^2) \frac{\partial}{\partial y}
  .\)
  </center>
  <p style="margin-bottom: 1em">
  Find the connection matrix \(\omega = [\omega_j^i]\) and the curvature
  matrix \(\Omega = [\Omega_j^i]\) relative to the orthonormal frame
  \(e_1, e_2\) of the Riemannian connection on the Poincare disk. (Hint:
  first find the dual frame \(\theta^1, \theta^2\). Then solve for
  \(\omega_j^i\) in (11.9).)
  </p>
  <h2 id="auto-16">12<span style="margin-left: 1em"></span>The Theorema Egregium Using Forms<span
      style="margin-left: 1em"></span></h2>
  <p>
  第8节我们用向量场证明了高斯绝妙定理。这一节我们重新证明这个定理，但是用的是微分形式。一个关键步骤是高斯曲率公式的微分形式类比的推导。高斯绝妙定理给了曲面高斯曲率的内蕴性质，只依赖于度量而与欧氏空间中曲面的嵌入无关。这一特点可以用来定义抽象黎曼2流形的高斯曲率。我们以庞加莱半平面为例计算高斯曲率。
  </p>
  <h3 id="auto-17">12.1<span style="margin-left: 1em"></span>The Gauss Curvature Equation<span style="margin-left: 1em"></span></h3>
  <p>
  The Gauss curvature equation (Theorem 8.1) for an oriented surface \(M\)
  in \(\mathbb{R}^3\) relates the curvature tensor to the shape operator.
  It has an analogue in terms of differential forms.
  </p>
  <p>
  Consider a smooth surface \(M\) in \(\mathbb{R}^3\) and a point \(p\) in
  \(M\). Let \(U\) be an open neighborhood of \(p\) in \(M\) on which
  there is an orthonormal frame \(e_1, e_2\). This is always possible by
  the Gram-Schimidt process, which turns any frame into an orthonormal
  frame. Let \(e_3\) be the cross product \(e_1 \times e_2\). Then \(e_1,
  e_2, e_3\) is an orthonormal frame for the vector bundle \(T\mathbb{R}^3
  |\_U\) over \(U\).
  </p>
  <p>
  For the connection \(D\) on the bundle \(T\mathbb{R}^3 |\_M\), let
  \([\omega_j^i]\) be the connection matrix of 1-forms relative to the
  orthonormal frame \(e_1, e_2, e_3\) over \(U\). Since \(D\) is
  compatible with the metric and the frame \(e_1, e_2, e_3\) is
  orthonormal, the matrix \([\omega_j^i]\) is skew-symmetric (Proposition
  11.4). Hence, for \(X \in \mathfrak{X} (M)\),
  </p>
  <center>
  \(\displaystyle \begin{array}{lllll}
  D_X e_1 = & & - \omega_2^1 (X)
  e_2 & - \omega_3^1 (X) e_3, & \hspace{3cm}  
   \text{(12.1)}\\
  D_X e_2
  = & \omega_2^1 (X) e_1 & & - \omega_3^2 (X) e*{3,} & \hspace{3cm}

        \text{(12.2)}\\

  D*X e_3 = & \omega_3^1 (X) e_1 & + \omega_3^2 (X) e_2
  . & & \hspace{3cm}  
   \text{(12.3)}
  \end{array}\)
  </center>
  <p>
  Let \(\nabla\) be the Riemannian connection on the surface \(M\). Recall
  that for \(X, Y \in \mathfrak{X} (M)\), the directional derivative \(D_X
  Y\) need not be tangent to the surface \(M\), and \(\nabla_X Y\) is
  simply the tangential component \((D_X Y)*{\tan}\) of \(D*X Y\). By
  (12.1) and (12.2),
  </p>
  <center>
  \(\displaystyle \begin{array}{lll}
  \nabla_X e_1 = (D_X e_1)*{\tan} = & - & \omega*2^1 (X) e_2,\\
  \nabla_X e_2 = (D_X e_2)*{\tan} = & &
  \omega^1_2 (X) e_1 .
  \end{array}\)
  </center>
  <p>
  It follows that the connection matrix of the Riemannian connection
  \(\nabla\) on \(M\) is
  </p>
  <center>
  \(\displaystyle \omega = \left[ \begin{array}{cc}
  0 & \omega_2^1\\

*       \omega_2^1 & 0

  \end{array} \right] = \left[ \begin{array}{cc}
  0 & 1\\

        - 1 & 0

  \end{array} \right] \omega_2^1 .\)
  </center>
  <p>
  Since
  </p>
  <center>
  \(\displaystyle \omega \wedge \omega = \left[ \begin{array}{cc}
  0 &
  \omega_2^1\\

* \omega_2^1 & 0
  \end{array} \right] \wedge \left[
  \begin{array}{cc}
  0 & \omega_2^1\\
* \omega_2^1 & 0
  \end{array}
  \right] = \left[ \begin{array}{cc}
* \omega_2^1 \wedge \omega_2^1 &
  0\\
  0 & - \omega_2^1 \wedge \omega_2^1
  \end{array} \right] = \left[
  \begin{array}{cc}
  0 & 0\\
  0 & 0
  \end{array} \right],\)
  </center>
  <p>
  the curvature matrix of \(\nabla\) is
  </p>
  <center>
  \(\displaystyle \Omega = d \omega + \omega \wedge \omega = d \omega =
  \left[ \begin{array}{cc}
  0 & d \omega_2^1\\
* d \omega_2^1 &
  0
  \end{array} \right] = \left[ \begin{array}{cc}
  0 & 1\\
* 1 &
  0
  \end{array} \right] d \omega_2^1 .\)
  </center>
  <p>
  So the curvature matrix of \(\nabla\) is completely described by
  </p>
  <table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle \Omega_2^1 = d \omega_2^1 .\)</td>
  <td align="right">(12.4)</td>
  </tr>
  </table>
  <p>
  Set the unit normal vector field \(N\) on \(U\) to be \(N = - e_3\). By
  (12.3), the shape operator \(L\) is described by
  </p>
  <table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle L (X) = - D_X N = D_X e_3 =
  \omega_3^1 (X) e_1 + \omega_3^2 (X) e_2, \quad X
  \in \mathfrak{X} (M)
  .\)</td>
  <td align="right">(12.5)</td>
  </tr>
  </table>
  <p style="margin-top: 1em">
  <strong>定理 <class style="font-style: normal">12.1</class>. </strong><i>(Gauss
  curvature equation). Let \(e_1, e_2\) be an orthonormal frame of vector
  fields on an oriented open subset \(U\) of a surface \(M\) in
  \(\mathbb{R}^3\), and let \(e_3\) be a unit normal vector field on
  \(U\). Relative to \(e_1, e_2, e_3\), the curvature form \(\Omega_2^1\)
  of the Riemannian connection on \(M\) is related to the connection forms
  of the directional derivative \(D\) on the bundle \(T\mathbb{R}^3 |\_M\)
  by</i>
  </p>
  <i><table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle \Omega_2^1 = \omega_3^1 \wedge
  \omega_3^2 .\)</td>
  <td align="right">(12.6)</td>
  </tr>
  </table></i>
  <p style="margin-bottom: 1em">
  <i>We call formula (12.6) the Gauss curvature equation, because on the
  left-hand side, \(\Omega_2^1\) describes the curvature tensor of the
  surface, while on the right-hand side, \(\omega_3^1\) and \(\omega_3^2\)
  describe the shape operator.</i>
  </p>
  <p style="margin-top: 1em">
  <strong>证明. </strong>Let \(\tilde{\Omega}\_j^i\) be the
  curvature forms of the connection \(D\) on \(T\mathbb{R}^3 |\_M\).
  Because the curvature tensor of \(D\) is zero, the secodn structural
  equation for \(\tilde{\Omega}\) gives
  </p>
  <table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle \tilde{\Omega}\_j^i = d \omega_j^i +
  \sum_k \omega_k^i \wedge \omega_j^k = 0.\)</td>
  <td align="right">(12.7)</td>
  </tr>
  </table>
  <p>
  In particular,
  </p>
  <center>
  \(\displaystyle d \omega_2^1 + \omega^1_1 \wedge \omega^1_2 + \omega^1_2
  \wedge \omega^2_2 +
  \omega^1_3 \wedge \omega^3_2 = 0.\)
  </center>
  <p>
  Since \(\omega^1_1 = \omega^2_2 = 0\), this reduces to
  </p>
  <center>
  \(\displaystyle d \omega^1_2 + \omega^1_3 \wedge \omega^3_2 = 0.\)
  </center>
  <p>
  Since the matrix \([\omega^i_j]\) is skew-symmetric,
  </p>
  <center>
  \(\displaystyle d \omega^1_2 = \omega^1_3 \wedge \omega^2_3 .\)
  </center>
  <p style="margin-bottom: 1em">
  The Gauss curvature equation now follows from (12.4).<span style="margin-left: 1em"></span>\(\Box\)
  </p>
  <h3 id="auto-18">12.2<span style="margin-left: 1em"></span>The Theorema Egregium<span style="margin-left: 1em"></span></h3>
  <p>
  We will derive formulas for the Gaussian curvature of a surface in
  \(\mathbb{R}^3\), first in terms of the connection forms for the
  directional derivative and then in terms of the curvature form
  \(\Omega^1_2\).
  </p>
  <p style="margin-top: 1em">
  <strong>命题 <class style="font-style: normal">12.2</class>. </strong><i>For a
  smooth surface in \(\mathbb{R}^3\), if \(e_1, e_2\) is an orthonormal
  frame over an oriented open subset \(U\) of the surface and \(e_3\) is a
  unit normal vector field on \(U\), then the Gaussian curvature \(K\) on
  \(U\) is given by</i>
  </p>
  <p style="margin-bottom: 1em">
  <i><center>
  \(\displaystyle K = \det \left[ \begin{array}{cc}
  \omega^1_3 (e_1) &
  \omega^1_3 (e_2)\\
  \omega^2_3 (e_1) & \omega^2_3 (e_2)
  \end{array}
  \right] = (\omega^1_3 \wedge \omega^2_3) (e_1, e_2) .\)
  </center></i>
  </p>
  <p style="margin-top: 1em">
  <strong>证明. </strong>From (12.5),
  </p>
  <center>
  \(\displaystyle \begin{array}{l}
  L (e_1) = \omega^1_3 (e_1) e_1 +
  \omega^2_3 (e_1) e_2,\\
  L (e_2) = \omega^1_3 (e_2) e_1 + \omega^2_3
  (e_2) e_2 .
  \end{array}\)
  </center>
  <p>
  So the matrix of \(L\) relative to the frame \(e_1, e_2\) is
  </p>
  <center>
  \(\displaystyle \left[ \begin{array}{cc}
  \omega^1_3 (e_1) & \omega^1_3
  (e_2)\\
  \omega^2_3 (e_1) & \omega^2_3 (e_2)
  \end{array} \right] .\)
  </center>
  <p>
  Therefore,
  </p>
  <center>
  \(\displaystyle \begin{array}{ll}
  K & = \det L = \det \left[
  \begin{array}{cc}
  \omega^1_3 (e_1) & \omega^1_3 (e_2)\\

        \omega^2_3 (e_1) & \omega^2_3 (e_2)

  \end{array} \right]\\
  & =
  \omega^1_3 (e_1) \omega^2_3 (e_2) - \omega^1_3 (e_2) \omega^2_3 (e_1) =

        \left( \omega^1_3 {\wedge \omega^2_3}  \right) (e_1, e_2)
        .

  \end{array}\)
  </center>
  <p style="margin-bottom: 1em">
  <span style="margin-left: 1em"></span>\(\Box\)
  </p>
  <p style="margin-top: 1em">
  <strong>定理 <class style="font-style: normal">12.3</class>.
  </strong><i>(Theorema Egregium). For a smooth surface in
  \(\mathbb{R}^3\), if \(e*1, e_2\) is an orthonormal frame over an open
  subset \(U\) of the surface with dual frame \(\theta^1, \theta^2\), then
  the Gaussian curvature \(K\) on \(U\) is given by</i>
  </p>
  <i><table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle K = \Omega^1_2 (e_1, e_2)\)</td>
  <td align="right">(12.8)</td>
  </tr>
  </table></i>
  <p>
  <i>or by </i>
  </p>
  <div style="margin-bottom: 1em">
  <i><table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle d \omega^1_2 = K \theta^1 \wedge
  \theta^2 .\)</td>
  <td align="right">(12.9)</td>
  </tr>
  </table></i>
  </div>
  <p style="margin-top: 1em">
  <strong>证明. </strong>Formula (12.8) is an immediate
  consequence of Proposition 12.2 and the Gauss curvature equation (12.6).
  </p>
  <p>
  As for (12.9), since
  </p>
  <center>
  \(\displaystyle K = K (\theta^1 \wedge \theta^2) (e_1, e_2) = \Omega^1_2
  (e_1, e_2)\)
  </center>
  <p>
  and a 2-form on \(U\) is completely determined by its value on \(e_1,
  e_2\), we have \(\Omega^1_2 = K \theta^1 \wedge \theta^2\). By (12.4),
  </p>
  <p style="margin-bottom: 1em">
  <center>
  \(\displaystyle d \omega^1_2 = K \theta^1 \wedge \theta^2 .\)
  </center>
  <span style="margin-left: 1em"></span>
  \(\Box\)
  </p>
  <p>
  By the definition of the curvature matrix, if \(e_1, e_2\) is an
  orthonormal frame on an open subset \(U\) of \(M\), then
  </p>
  <center>
  \(\displaystyle R (X, Y) e_2 = \Omega^1_2 (X, Y) e_1
  \quad
  \operatorname{for}\operatorname{all}X, Y \in \mathfrak{X} (U) .\)
  </center>
  <p>
  so that
  </p>
  <center>
  \(\displaystyle \langle R (e_1, e_2) e_2, e_1 \rangle = \langle
  \Omega^1_2 (e_1, e_2) e_1, e_1
  \rangle = \Omega^1_2 (e_1, e_2) .\)
  </center>
  <p style="margin-top: 1em">
  <strong>定义 <class style="font-style: normal">12.4</class>. </strong><i>The
  <b><i>Gaussian curvature</i></b> K at a point \(p\) of a Riemannian
  2-manifold \(M\) is defined to be</i>
  </p>
  <i><table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle K_p = \langle R_p (u, v) v, u
  \rangle\)</td>
  <td align="right">(12.10)</td>
  </tr>
  </table></i>
  <p style="margin-bottom: 1em">
  <i>for any orthonormal basis \(u, v\) for the tangent plane \(T_p
  M\).</i>
  </p>
  <p>
  为了让高斯曲率定义有效，我们需要证明公式(12.10)不依赖于正交基的选择。我们在下一小节证明。
  </p>
  <h3 id="auto-19">12.3<span style="margin-left: 1em"></span>Skew-Symmetric of the Curvature
  Tensor<span style="margin-left: 1em"></span></h3>
  <p>
  Recall that the curvature of an affine connection \(\nabla\) on a
  manifold \(M\) is defined to be
  </p>
  <center>
  \(\displaystyle R : \mathfrak{X} (M) \times \mathfrak{X} (M) \times
  \mathfrak{X} (M)
  \rightarrow \mathfrak{X} (M),\)
  </center>
  <center>
  \(\displaystyle R (X, Y) Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z -
  \nabla*{[X, Y]} Z.\)
  </center>
  <p>
  We showed that \(R (X, Y) Z\) is \(\mathcal{F}\)-linear in every
  argument; therefore, it is a point operator. A point operator is also
  called a <b><i>tensor</i></b>. It is immediate from the definition that
  the curvature tensor \(R (X, Y) Z\) is skew-symmetric in \(X\) and
  \(Y\).
  </p>
  <p style="margin-top: 1em; margin-bottom: 1em">
  <strong>命题 <class style="font-style: normal">12.5</class>. </strong><i>If an
  affine connection \(\nabla\) on a Riemannian manifold \(M\) is
  compatible with the metric, then for vector fields \(X, Y, Z, W \in
  \mathfrak{X} (M)\), the tensor \(\langle R (X, Y) Z, W \rangle\) is
  skew-symmetric in \(Z\) and \(W\).</i>
  </p>
  <p>
  We now show that \(\langle R (u, v) v, u \rangle\) is independent of the
  orthonormal basis \(u, v\) for \(T*p M\). Suppose \(\bar{u}, \bar{v}\)
  is another orthonormal basis. Then
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  \bar{u} & = a u + b v,\\
  \bar{v} &
  = c u + d v
  \end{array}\)
  </center>
  <p>
  for an orthonormal matrix \(A = \left[ \begin{array}{cc}
  a & b\\
  c &
  d
  \end{array} \right]\), and
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  \langle R (\bar{u}, \bar{v})
  \bar{v}, \bar{u} \rangle & = \langle (\det A) R
  (u, v) (c u + d v), a
  u + b v \rangle\\
  & = (\det A)^2 \langle R (u, v) v, u
  \rangle
  \end{array}\)
  </center>
  <p>
  by the skew-symmetry of \(\langle R (u, v) z, w \rangle\) in \(z\) and
  \(w\). Since \(A \in O (2)\), \(\det A = \pm 1\). Hence,
  </p>
  <center>
  \(\displaystyle \langle R (\bar{u}, \bar{v}) \bar{v}, \bar{u} \rangle =
  \langle R (u, v) v, u
  \rangle .\)
  </center>
  <h3 id="auto-20">12.4<span style="margin-left: 1em"></span>Sectional Curvarture<span style="margin-left: 1em"></span></h3>
  <p>
  Let \(M\) be a Riemannian manifold and \(p\) a point in \(M\). If \(P\)
  is a 2-dimensional subspace of the tangent spacce \(T_p M\), then we
  define the <b><i>sectional curvature</i></b> of \(P\) to be
  </p>
  <table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle K (P) = \langle R (e_1, e_2) e_2,
  e_1 \rangle\)</td>
  <td align="right">(12.11)</td>
  </tr>
  </table>
  <p>
  for any orthonormal basis \(e_1, e_2\) of \(P\). Just as in the
  definition of the Gaussian curvature, the right-hand side of (12.11) is
  independent of the orthonormal basis \(e_1, e_2\).
  </p>
  <p>
  If \(u, v\) is an arbitrary basis for the 2-plane \(P\), then a
  computation similar to that in Section 8.3 shows that the sectional
  curvature of \(P\) is also given by
  </p>
  <center>
  \(\displaystyle K (P) = \frac{\langle R (u, v) v, u \rangle}{\langle u,
  u \rangle \langle v, v
  \rangle - \langle u, v \rangle^2} .\)
  </center>
  <h3 id="auto-21">12.5<span style="margin-left: 1em"></span>Poincar&eacute; Half-Plane<span style="margin-left: 1em"></span></h3>
  <p style="margin-top: 1em">
  <strong>例 <class style="font-style: normal">12.6</class>. </strong>(The Gaussian
  curvature of the Poincar&eacute; half-plane). The Poincar&eacute;
  half-plane is the upper half-plane
  </p>
  <center>
  \(\displaystyle \mathbb{H}^2 = \{ (x, y) \in \mathbb{R}^2 |y > 0 \}\)
  </center>
  <p>
  with the metric
  </p>
  <center>
  \(\displaystyle \langle, \rangle*{(x, y)} = \frac{d x \otimes d x + d y
  \otimes d y}{y^2} .\)
  </center>
  <p>
  Classically, the notation for a Riemannian metric is \(d s^2\). Hence,
  the metric on the Poincar&eacute; half-plane is
  </p>
  <center>
  \(\displaystyle d s^2 = \frac{d x \otimes d x + d y \otimes d y}{y^2}
  .\)
  </center>
  <p>
  With this metric, an orthonormal frame is
  </p>
  <center>
  \(\displaystyle e_1 = y \frac{\partial}{\partial x}, \quad e_2 = y
  \frac{\partial}{\partial y}
  .\)
  </center>
  <p>
  So the dual frame is
  </p>
  <center>
  \(\displaystyle \theta^1 = \frac{1}{y} d x, \quad \theta^2 = \frac{1}{y}
  d y.\)
  </center>
  <p>
  Hence,
  </p>
  <table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle d \theta^1 = \frac{1}{y^2} d x
  \wedge d y, \quad d \theta^2 = 0.\)</td>
  <td align="right">(12.12)</td>
  </tr>
  </table>
  <p>
  On the Poincare half-plane the connection form \(\omega^1_2\) is a
  linear combination of \(d x\) and \(d y\), so we may write
  </p>
  <table width="100%">
  <tr>
  <td width="100%" align="center">\(\displaystyle \omega^1_2 = a d x + b d y.\)</td>
  <td align="right">(12.13)</td>
  </tr>
  </table>
  <p>
  We will determine the coefficients \(a, b\) from the first structural
  equation:
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  d \theta^1 = & - \omega^1_2 \wedge
  \theta^2, \hspace{3cm} \text{(12.14)}\\
  d \theta^2 = & - \omega^2_1
  \wedge \theta^1 = \omega^1_2 \wedge \theta^1 .
  \hspace{3cm}  
   \text{(12.15)}
  \end{array}\)
  </center>
  <p>
  By (12.12), (12.13) and (12.14),
  </p>
  <center>
  \(\displaystyle \frac{1}{y^2} d x \wedge d y = d \theta^1 = - (a d x + b
  d y) \wedge
  \frac{1}{y} d y = - \frac{a}{y} d x \wedge d y.\)
  </center>
  <p>
  So \(a = - 1 / y\). By (12.12), (12.13) and (12.15),
  </p>
  <center>
  \(\displaystyle 0 = d \theta^2 = \left( - \frac{1}{y} d x + b d y
  \right) \wedge \frac{1}{y} d
  x = - \frac{b}{y} d x \wedge d y.\)
  </center>
  <p>
  So \(b = 0\). Therefore,
  </p>
  <center>
  \(\displaystyle \begin{array}{rl}
  \omega^1_2 & = - \frac{1}{y} d x,\\

        d \omega^1_2 & = \frac{1}{y^2} d y \wedge d x\\

  & = - \frac{1}{y^2} d
  x \wedge d y.
  \end{array}\)
  </center>
  <p>
  By definition, the Gaussian curvature of the Poincare half-plane is
  </p>
  <p style="margin-bottom: 1em">
  <center>
  \(\displaystyle \begin{array}{rl}
  K & = \Omega^1_2 (e_1, e_2) = -
  \frac{1}{y^2} (d x \wedge d y) \left( y
  \frac{\partial}{\partial x},
  y \frac{\partial}{\partial y} \right)\\
  & = - (d x \wedge d y)
  \left( \frac{\partial}{\partial x},
  \frac{\partial}{\partial y}
  \right) = - 1.
  \end{array}\)
  </center>
  </p>
  <p style="margin-top: 1em">
  <strong>例. </strong>The orthogonal group \(O (2)\)
  </p>
  <ol>
  <li>
  <p>
  Show that an element \(A\) of \(O (2)\) is either
  </p>
  <center>
  \(\displaystyle A = \left[ \begin{array}{cc}
  a & - b\\
  b &
  a
  \end{array} \right] \quad \operatorname{or} \quad \left[
  \begin{array}{cc}
  a & b\\
  b & - a
  \end{array} \right],\)
  </center>
  <p>
  where \(a^2 + b^2 = 1, a, b \in \mathbb{R}\).
  </p>
  </li>
  <li>
  <p>
  Let \(\operatorname{SO} (2) = \{ A \in O (2) | \det A = 1 \}\). Show
  that every element \(A\) of SO(2) is of the form
  </p>
  <center>
  \(\displaystyle A = \left[ \begin{array}{cc}
  \cos t & - \sin t\\

            \sin t & \cos t

  \end{array} \right], \quad t \in \mathbb{R}.\)
  </center>
  <p>
  Thus, \(\operatorname{SO} (2)\) is the group of rotations about the
  origin in \(\mathbb{R}^2\).
  </p>
  </li>
  </ol>
  <p style="margin-bottom: 1em">
  Let \(J = \left[ \begin{array}{cc}
  1 & 0\\
  0 & - 1
  \end{array}
  \right]\). Then \(O (2) =\operatorname{SO} (2) \cup \operatorname{SO}
  (2) J\). This proves that \(O (2)\) has two connected components, each
  diffeomorphic to the circle \(S^1\).
  </p>
    </body>
