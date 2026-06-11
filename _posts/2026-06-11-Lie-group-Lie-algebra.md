---
layout: post
title: notes of Lie Group and Lie Algebra
date: 2026-06-11 17:00:00
description: notes of Chap 4 of The Introduction to Manifolds, Loring W. Tu
tags: Lie group
categories: math
tikzjax: true
---

<body>
    <h2 id="auto-1">1<span style="margin-left: 1em"></span>Bump Functions and Partitions of Unity<span
    style="margin-left: 1em"></span></h2>
    <p>
      A partition of unity on a manifold is a collection of nonnegative
      functions that sum to 1.
      (这不是一种概率测度吗？)
      Usually one demands in addition that the partition of unity be
      subordinate to an open cover \(\{ U_{\alpha} \}_{\alpha \in A}\). What
      this means is that the partition of unity \(\{ \rho_{\alpha} \}_{\alpha
      \in A}\) is indexed by the same set as the open cover and for each
      \(\alpha\) in the index \(A\), the support of \(\rho_{\alpha}\) is
      contained in \(U_{\alpha}\). In particular, \(\rho_{\alpha}\) vanishes
      outside \(U_{\alpha}\).
    </p>
    <p>
      It is the single feature that makes the behavior of \(C^{\infty}\)
      manifolds so different from that of real-analytic or complex manifolds.
      We prove the existence of a \(C^{\infty}\) partition of unity on a
      compact manifold. The proof of the existence of a \(C^{\infty}\)
      partition of unity on a general manifold is more technical and is
      postponed to appendix.
    </p>
    <p>
      A partition of unity is used in two ways: (1) to decompose a global
      object on a manifold into a locally finite sum of local objects on the
      open sets \(U_{\alpha}\) of an open cover, and (2) to patch together
      local objects on the open sets \(U_{\alpha}\) into a global objet on the
      manifold. Thus, a partition of unity serves as a bridge between global
      and local analysis on a manifold, there may be no global coordinates. 
    </p>
    <h3 id="auto-2">1.1<span style="margin-left: 1em"></span>\(C^{\infty}\) Bump Functions<span style="margin-left: 1em"></span><span
    style="margin-left: 1em"></span></h3>
    <p>
      Recall that \(\mathbb{R}^{\times}\) denotes the set of nonzero real
      numbers. The support of a real-valued function \(f\) on a manifold \(M\)
      is defined to be the closure in \(M\) of the subset on which \(f \neq
      0\)
    </p>
    <center>
      \(\displaystyle \operatorname{supp}f =\operatorname{cl}_M (f^{- 1}
      (\mathbb{R}^{\times}))
=\operatorname{closure}\operatorname{of} \{ q \in
      M|f (q) \neq 0 \}
\operatorname{in}M.\)
    </center>
    <center>
      \(\displaystyle f (t) = \left\{ \begin{array}{ll}
  e^{- 1 / t} &
      \operatorname{for}t > 0,\\
  0 & \operatorname{for}t \leq 0
\end{array}
      \right.\)
    </center>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle g (t) = \frac{f (t)}{f (t) + f (1 -
        t)} n\)</td>
        <td align="right">(1.1)</td>
      </tr>
    </table>
    <p>
      Given two positive real numbers \(a < b\), we make a linear change of
      variables to map \([a^2, b^2]\) to \([0, 1]\). Let 
    </p>
    <center>
      \(\displaystyle h (x) = g \left( \frac{x - a^2}{b^2 - a^2} \right)\)
    </center>
    <p>
      Replace \(x\) by \(x^2\) to make the function symmetric in \(x\): \(k
      (x) = h (x^2)\).
    </p>
    <p>
      Finally, set 
    </p>
    <center>
      \(\displaystyle \rho (x) = 1 - k (x) = 1 - g \left( \frac{x^2 - a^2}{b^2
      - a^2} \right) .\)
    </center>
    <p>
      The \(\rho (x)\) is a \(C^{\infty}\) function at 0 at \(\mathbb{R}\)
      that is identically 1 on \([- a, a]\) and has support in \([- b, b]\).
    </p>
    <p>
      It is easy to extend the construction of a bump function from
      \(\mathbb{R}\) to \(\mathbb{R}^n\). To get a \(C^{\infty}\) bump
      function at \(\boldsymbol{0}\) in \(\mathbb{R}^n\) that is 1 on the
      closed ball \(\bar{B} (\boldsymbol{0}, a)\) and has support in the
      closed ball \(\bar{B} (\boldsymbol{0}, b)\), set
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \rho (x) = \rho (\| x \|) = 1 - g
        \left( \frac{\| x \| r^2 - a^2}{b^2 - a^2}
\right) .\)</td>
        <td align="right">(1.2)</td>
      </tr>
    </table>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.1</class>. </strong>(Bump function
      supported in an open set).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1.2</class>.
      </strong><i>(\(C^{\infty}\) extension of a function). Suppose \(f\) is a
      \(C^{\infty}\) function defined on a neighborhood \(U\) of a point \(p\)
      in a manifold \(M\). Then there is a \(C^{\infty}\) function
      \(\tilde{f}\) on \(M\) that agrees with \(f\) in some possibly smaller
      neighborhood of \(p\).</i>
    </p>
    <h3 id="auto-3">1.2<span style="margin-left: 1em"></span>Partitions of Unity<span style="margin-left: 1em"></span></h3>
    <p>
      If \(\{ U_i \}_{i \in I}\) is a finte open cover of \(M\), a
      \(C^{\infty}\) partition of unity subordinate to \(\{ U_i \}_{i \in I}\)
      is a collection of nonnegative \(C^{\infty}\) functions \(\{ \rho_i : M
      \rightarrow \mathbb{R} \}_{i \in I}\) such that \(\operatorname{supp}
      \rho_i \subset U_i\) and 
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \sum \rho_i = 1.\)</td>
        <td align="right">(1.3)</td>
      </tr>
    </table>
    <p>
      When \(I\) is an infinite set, we will impose a local finiteness
      condition.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.3</class>. </strong>(An open cover that
      is not locally finite). Let \(U_{r, n}\) be the open interval \(\left] r
      - \frac{1}{n}, r + \frac{1}{n} \right[\) on the real line
      \(\mathbb{R}\). The open cover \(\{ U_{r, n} |r \in \mathbb{Q}, n \in
      \mathbb{Z}^+ \}\) of \(\mathbb{R}\) is not locally finite.
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">1.4</class>. </strong><i>A
      \(C^{\infty}\) partition of unity on a manifold is a collection of
      nonnegative \(C^{\infty}\) functions \(\{ \rho_{\alpha} : M \rightarrow
      R \}\) such that</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            the collection of supports, \(\{ \operatorname{supp} \rho_{\alpha}
            \}_{\alpha \in A}\) is locally finite,
          </p>
        </li>
        <li>
          <p>
            \(\sum \rho_{\alpha} = 1.\)
          </p>
        </li>
      </ol></i>
    </div>
    <h3 id="auto-4">1.3<span style="margin-left: 1em"></span>Existence of a Partition of Unity<span
    style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em">
      <strong>引理 <class style="font-style: normal">1.5</class>. </strong><i>If
      \(\rho_1, \ldots, \rho_m\) are real-valued functions on a manifold
      \(M\), then </i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \operatorname{supp} \left( \sum \rho_i \right) \subset
        \bigcup
\operatorname{supp} \rho_i .\)
      </center></i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1.6</class>. </strong><i>Let
      \(M\) be a compact manifold and \(\{ U_{\alpha} \}_{\alpha \in A}\) an
      open cover of \(M\). There exists a \(C^{\infty}\) partition of unity
      \(\{ \rho_{\alpha} \}_{\alpha \in A}\) subordinate to \(\{ U_{\alpha}
      \}_{\alpha \in A}\).</i>
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">1.7</class>.
      </strong><i>(Existence of a \(C^{\infty}\) partition of unity). Let \(\{
      U_{\alpha} \}_{\alpha \in A}\) be an open cover of a manifold \(M\).</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            There is a \(C^{\infty}\) partition of unity \(\{ \varphi_k \}_{k
            = 1}^{\infty}\) with every \(\varphi_k\) having compact support
            such that for each \(k\), \(\operatorname{supp} \varphi_k \subset
            U_{\alpha}\) for some \(\alpha \in A\).
          </p>
        </li>
        <li>
          <p>
            If we do not require compact support, then there is a
            \(C^{\infty}\) partition of unity \(\{ \rho_{\alpha} \}\)
            subordinate to \(\{ U_{\alpha} \}\).
          </p>
        </li>
      </ol></i>
    </div>
    <h2 id="auto-5">2<span style="margin-left: 1em"></span>Velocity Fields<span style="margin-left: 1em"></span></h2>
    <p>
      A vector field \(X\) on a manifold \(M\) is the assignment of a tangent
      vector \(X_p \in T_p M\) to each point \(p \in M\). More formally, a
      vector field on \(M\) is a section of the tangent bundle \(T M\) of
      \(M\). In the first subsection, we give two other characterizations of
      smooth vector fields, in terms of the coefficients relative to
      coordinate vector fields and in terms of smooth functions on the
      manifold.
    </p>
    <p>
      Every vector field may be viewer locally as the velocity vector field of
      a fluid view. The path traced out by a point under this flow is called
      an integral curve of the vector field. Finding the equation of an
      integral curve is equivalent to solving a system of first-order ordinary
      differential equations (ODE). Thus, the theorey of ODE guarantees the
      existence of integral curves.
    </p>
    <p>
      The set \(\mathfrak{X} (M)\) of all \(C^{\infty}\) vector fields on a
      manifold \(M\) clearly has the structure of a vector space. We introduce
      a bracket operation \([,]\) that makes it into a Lie algebra. Because
      vector fields do not push forward under smooth maps, the Lie algebra
      \(\mathfrak{X} (M)\) does not give rise to a functor on the category of
      smooth manifolds. Nonetheless, there is a notion of <i>related vector
      fields</i> that allows us to compare vector fields on two manifolds
      under a smooth map.
    </p>
    <h3 id="auto-6">2.1<span style="margin-left: 1em"></span>Smoothness of a Vector Field<span style="margin-left: 1em"></span></h3>
    <p>
      We defined a vector field \(X\) on a manifold \(M\) to be smooth if the
      map \(X : M \rightarrow T M\) is smooth as a section of the tangent
      bundle \(\pi : T M \rightarrow M\). In a coordinate chart \((U, \phi) =
      (U, x^1, \ldots, x^n)\) on \(M\), the value of the vector field \(X\) at
      \(p \in U\) is a linear combination
    </p>
    <center>
      \(\displaystyle X_p = \sum a^i (p) \frac{\partial}{\partial x^i} |_p .\)
    </center>
    <p>
      As \(p\) varies in \(U\), the coefficients \(a^i\) become functions on
      \(U\).
    </p>
    <p>
      The chart \((U, \phi) = (U, x^1, \ldots, x^n)\) on the manifold \(M\)
      induces a chart
    </p>
    <center>
      \(\displaystyle (T U, \tilde{\phi}) = (T U, \bar{x}^1, \ldots,
      \bar{x}^n, c^1, \ldots, c^n)\)
    </center>
    <p>
      on the tangent bundle \(T M\), where \(\bar{x}^i = \pi^{\ast} x^i = x^i
      \circ \pi\) and \(c^i\) are defined by
    </p>
    <center>
      \(\displaystyle v = \sum c^i (v) \frac{\partial}{\partial x^i} |_p,
      \quad v \in T_p M.\)
    </center>
    <p>
      Comparing coefficients in 
    </p>
    <center>
      \(\displaystyle X_p = \sum a^i (p) \frac{\partial}{\partial x^i} |_p =
      \sum c^i (X_p)
\frac{\partial}{\partial x^i} |_p, \quad p \in U,\)
    </center>
    <p>
      we get \(a^i = c^i \circ X\) as functions on \(U\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>引理 <class style="font-style: normal">2.1</class>.
      </strong><i>(Smoothness of a vector field on a chart). Let \((U, \phi)\)
      be a chart on a manifold \(M\). A vector field \(X = \sum a^i \partial /
      \partial x^i\) on \(U\) is smooth if and only if the coefficient
      functions \(a^i\) are all smooth on \(U\).</i>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">2.2</class>.
      </strong><i>(Smoothness of a vector field in terms of coefficients). Let
      \(X\) be a vector field on a manifold \(M\). The following are
      equivalent:</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            The vector field \(X\) is smooth on \(M\).
          </p>
        </li>
        <li>
          <p>
            The manifold \(M\) has an atlas such that on any chart \((U, \phi)
            = (U, x^1, \ldots, x^n)\) of the atlas, the coefficients \(a^i\)
            of \(X = \sum a^i \partial / \partial x^i\) relative to the frame
            \(\partial / \partial x^i\) are all smooth.
          </p>
        </li>
        <li>
          <p>
            On any chart \((U, \phi) = (U, x^1, \ldots, x^n)\) on the manifold
            \(M\), the coefficients \(a^i\) of \(X = \sum a^i \partial /
            \partial x^i\) relative to the frame \(\partial / \partial x^i\)
            are all smooth.
          </p>
        </li>
      </ol></i>
    </div>
    <p>
      A vector field \(X\) on a manifold \(M\) induces a linear map on the
      algebra \(C^{\infty} (M)\) of \(C^{\infty}\) functions on \(M\): for \(f
      \in C^{\infty} (M)\), define \(X f\) to be the function
    </p>
    <center>
      \(\displaystyle (X f) (p) = X_p f, \quad p \in M.\)
    </center>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">2.3</class>.
      </strong><i>(Smoothness of a vector field in terms of functions). A
      vector field \(X\) on \(M\) is smooth if and only if for every smooth
      function \(f\) on \(M\), the function \(X f\) is smooth on \(M\).</i>
    </p>
    <p>
      By Proposition 2.3, we may view a \(C^{\infty}\) vector field \(X\) as a
      linear operator \(X : C^{\infty} (M) \rightarrow C^{\infty} (M)\) on the
      algebra of \(C^{\infty}\) functions on \(M\). The linear operator \(X :
      C^{\infty} (M) \rightarrow C^{\infty} (M)\) is a derivation: for all
      \(f, g \in C^{\infty} (M)\),
    </p>
    <center>
      \(\displaystyle X (f g) = (X f) g + f (X g) .\)
    </center>
    <p>
      In the following we think of \(C^{\infty}\) vector fields on \(M\)
      alternately as \(C^{\infty}\) sections of the tangent bundle \(T M\) and
      as derivations on the algebra \(C^{\infty} (M)\) of \(C^{\infty}\)
      functions. In fact, it can be shown that these two descriptions of
      \(C^{\infty}\) vector fields are equivalent.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">2.4</class>.
      </strong><i>(\(C^{\infty}\) extension of a vector field). Suppose \(X\)
      is a \(C^{\infty}\) vector field defined on a neighborhood \(U\) of a
      point \(p\) in a manifold \(M\). Then there is a \(C^{\infty}\) vector
      field \(\tilde{X}\) on \(M\) that agrees with \(X\) on some possibly
      smaller neighborhood of \(p\).</i>
    </p>
    <h3 id="auto-7">2.2<span style="margin-left: 1em"></span>Integral Curves<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">2.5</class>. </strong><i>Let
      \(X\) be a \(C^{\infty}\) vector field on a manifold \(M\), and \(p \in
      M\). An integral curve of \(X\) is a smooth curve \(c :] a, b
      [\rightarrow M\) such that \(c' (t) = X_{c (t)}\) for all \(t \in] a, b
      [\). Usually we assume that the open interval \(] a, b [\) contains 0.
      In this case, if \(c (0) = p\), then we say that \(c\) is an integral
      curve starting at \(p\) and call \(p\) the initial point of \(c\). To
      show the dependence of such an integral curve on the initial point
      \(p\), we also write \(c_t (p)\) instead of \(c (t)\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">2.6</class>. </strong><i>An
      integral curve is maximal if its domain cannot be extended to a larger
      interval.</i>
    </p>
    <p style="margin-top: 1em">
      <strong>例. </strong>Recall the vector field \(X_{(x, y)} =
      \langle - y, x \rangle\) on \(\mathbb{R}^2\). We will find an integral
      curve \(c (t)\) of \(X\) starting at the point \((1, 0) \in
      \mathbb{R}^2\). The condition for \(c (t) = (x (t), y (t))\) to be an
      integral curve is \(c' (t) = X_{c (t)}\), or
    </p>
    <center>
      \(\displaystyle \left[ \begin{array}{c}
  \dot{x} (t)\\
  \dot{y}
      (t)
\end{array} \right] = \left[ \begin{array}{c}
  - y (t)\\
  x
      (t)
\end{array} \right],\)
    </center>
    <p>
      so we need to solve the system of first-order ordinary differential
      equations
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \dot{x} = - y,\)</td>
        <td align="right">(2.1)</td>
      </tr>
    </table>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \dot{y} = x,\)</td>
        <td align="right">(2.2)</td>
      </tr>
    </table>
    <p>
      with initial condition \((x (0), y (0)) = (1, 0)\). From (2.1), \(y = -
      \dot{x}\), so \(\dot{y} = - \ddot{x}\). Substituting into (2.2) gives 
    </p>
    <center>
      \(\displaystyle \ddot{x} = - x.\)
    </center>
    <p>
      It is well known that the general solution to this equation is 
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle x = A \cos t + B \sin t.\)</td>
        <td align="right">(2.3)</td>
      </tr>
    </table>
    <p>
      Hence,
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle y = - \dot{x} = A \sin t - B \cos
        t.\)</td>
        <td align="right">(2.4)</td>
      </tr>
    </table>
    <p>
      The initial condition forces \(A = 1, B = 0\), so the integral curve
      starting at \((1, 0)\) is \(c (t) = (\sin t, \cos t)\), which
      parametrizes the unit circle.
    </p>
    <p>
      More generally, if intial point is \(p = (x_0, y_0)\) then 
    </p>
    <center>
      \(\displaystyle A = x_0, \quad B = y_0,\)
    </center>
    <p>
      and the general solution is 
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  x & = x_0 \cos t - y_0 \sin t\\
  y
      & = x_0 \sin t + y_0 \cos t, \quad t \in \mathbb{R}
\end{array}\)
    </center>
    <p>
      In matrix notation as 
    </p>
    <center>
      \(\displaystyle c (t) = \left[ \begin{array}{c}
  x (t)\\
  y
      (t)
\end{array} \right] = \left[ \begin{array}{cc}
  \cos t & - \sin
      t\\
  \sin t & \cos t
\end{array} \right] \left[ \begin{array}{c}
 
      x_0\\
  y_0
\end{array} \right] = \left[ \begin{array}{cc}
  \cos t & -
      \sin t\\
  \sin t & \cos t
\end{array} \right] p.\)
    </center>
    <p>
      Notice that 
    </p>
    <center>
      \(\displaystyle c_s (c_t (p)) = c_{s + t} (p) .\)
    </center>
    <p>
      For each \(t \in \mathbb{R}\),\(c_t : \mathbb{R}^2 \rightarrow
      \mathbb{R}^2\) is a diffeomorphism with inverse \(c_{- t}\).
    </p>
    <p style="margin-bottom: 1em">
      Let \(\operatorname{Diff} (M)\) be the group of diffeomorphisms of a
      manifold \(M\) with itself, the group operation being composition. A
      homomorphism \(c : \mathbb{R} \rightarrow \operatorname{Diff} (M)\) is
      called a <i>one-paramter group of diffeomorphisms</i> of \(M\). In this
      example the integral curves of the vector field \(X_{(x, y)} = \langle -
      y, x \rangle\) on \(\mathbb{R}^2\) give rise to a one-parameter group of
      diffeomorphisms of \(\mathbb{R}^2\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>Let \(X\) be the vector field \(x^2 d / d x\)
      on the real line \(\mathbb{R}\). Find the maximal integral curve of
      \(X\) starting at \(x = 2\).
    </p>
    <div style="margin-top: 0.5em; margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%"><strong>解答. </strong>Denote the integral
        curve by \(x (t)\). Then</font>
      </p>
    </div>
    <div style="margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%"><center>
          \(\displaystyle x' (t) = X_{x (t)} \Longleftrightarrow \dot{x} (t)
          \frac{d}{d x} = x^2
\frac{d}{d x}\)
        </center></font>
      </p>
    </div>
    <div style="margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%">where \(x' (t)\) is the velocity vector of the curve \(x
        (t)\), and \(\dot{x} (t)\) is the calculus derivative of the
        real-valued function \(x (t)\). Thus, \(x (t)\) satisfies the
        differential equaition</font>
      </p>
    </div>
    <div style="margin-left: 35.145870328812px">
      <font style="font-size: 84.1%"><table width="100%">
        <tr>
          <td width="100%" align="center">\(\displaystyle \frac{d x}{d t} = x^2, \quad x_0 =
          2.\)</td>
          <td align="right">(2.5)</td>
        </tr>
      </table></font>
    </div>
    <div style="margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%">By separation of variables:</font>
      </p>
    </div>
    <div style="margin-left: 35.145870328812px">
      <font style="font-size: 84.1%"><table width="100%">
        <tr>
          <td width="100%" align="center">\(\displaystyle \frac{d x}{x^2} = d t.\)</td>
          <td align="right">(2.6)</td>
        </tr>
      </table></font>
    </div>
    <div style="margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%">Integrating both sides gives </font>
      </p>
    </div>
    <div style="margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%"><center>
          \(\displaystyle - \frac{1}{x} = t + C, \operatorname{or} \quad x = -
          \frac{1}{t + C}\)
        </center></font>
      </p>
    </div>
    <div style="margin-bottom: 0.5em; margin-left: 35.145870328812px">
      <p>
        <font style="font-size: 84.1%">for some constant \(C\). \(C = - 1 / 2\). \(x (t) = 2 /
        (1 - 2 t)\).</font>
      </p>
    </div>
    <h3 id="auto-8">2.3<span style="margin-left: 1em"></span>Local Flows<span style="margin-left: 1em"></span></h3>
    <p>
      In general, if \(X\) is a smooth vector field on a manifold \(M\), to
      find an integral curve \(c (t)\) of \(X\) starting at \(p\), we first
      choose a coordinate chart \((U, \phi)\) about \(p\). In terms of the
      local coordinates,
    </p>
    <center>
      \(\displaystyle X_{c (t)} = \sum a^i (c (t)) \frac{\partial}{\partial
      x^i} |_{c (t)},\)
    </center>
    <p>
      and 
    </p>
    <center>
      \(\displaystyle c' (t) = \sum \dot{c}^i (t) \frac{\partial}{\partial
      x^i} |_{c (t)},\)
    </center>
    <p>
      where \(c^i (t) = x^i \circ c (t)\).<span style="margin-left: 1em"></span>The condition
      \(c' (t) = X_{c (t)}\) is thus equivalent to 
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \dot{c}^i (t) = a^i (c (t)) \quad
        \operatorname{for}i = 1, \ldots, n.\)</td>
        <td align="right">(2.7)</td>
      </tr>
    </table>
    <p>
      This is a system of ODE; the initial condition \(c (0) = p\) translates
      to \((c^1 (0), \ldots, c^n (0)) = (p^1, \ldots, p^n)\). By the existence
      and uniqueness theorem from the theorey of ODE, such a system always has
      a unique solution in the following sense.
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">2.7</class>. </strong><i>Let
      \(V\) be an open subset of \(\mathbb{R}^n\), \(p_0\) a point in \(V\),
      and \(f : V \rightarrow \mathbb{R}^n\) a \(C^{\infty}\) function. Then
      the differential equation</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle d y / d t = f (y), \quad y (0) = 0_0,\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>has a unique \(C^{\infty}\) solution \(y :] a (p_0), b (p_0)
      [\rightarrow V\), where \(] a (p_0), b (p_0) [\) is the maximal open
      interval containing \(0\) on which \(y\) is defined.</i>
    </p>
    <p>
      Next we would like to study the dependence of an integral curve on its
      initial point. Again we study the problem locally on \(\mathbb{R}^n\)
      The function \(y\) will now be a function of two arguments \(t\) and
      \(q\), and the condition for \(y\) to be an integral curve starting at
      the point \(q\) is
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \frac{\partial y}{\partial t} (t, q)
        = f (y (t, q)), \quad y (0, q) = q.\)</td>
        <td align="right">(2.8)</td>
      </tr>
    </table>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">2.8</class>. </strong><i>Let
      \(V\) be an open subset of \(\mathbb{R}^n\) and \(f : V \rightarrow
      \mathbb{R}^n\) a \(C^{\infty}\) function on \(V\). For each point \(p_0
      \in V\), there are a neighborhood \(W\) of \(p_0\) in \(V\), a number
      \(\varepsilon > 0\), and a \(C^{\infty}\) function</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle y :] - \varepsilon, \varepsilon [\times W \rightarrow
        V\)
      </center></i>
    </p>
    <p>
      <i>such that </i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \frac{\partial y}{\partial t} (t, q) = f (y (t, q)),
        \quad y (0, q) = q\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>for all \((t, q) \in] - \varepsilon, \varepsilon [\times W\).</i>
    </p>
    <p>
      It follows from Theorem 2.8 and (2.8) that if \(X\) is any
      \(C^{\infty}\) vector field on a chart \(U\) and \(p \in U\), then there
      are a neighborhood \(W\) of \(p\) in \(U\), an \(\varepsilon > 0\), and
      a \(C^{\infty}\) map
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle F :] - \varepsilon, \varepsilon
        [\times W \rightarrow U\)</td>
        <td align="right">(2.9)</td>
      </tr>
    </table>
    <p>
      such that for each \(q \in W\), the function \(F (t, q)\) is an integral
      curve of \(X\) starting at \(q\). In particular, \(F (0, q) = q\). We
      usually write \(F_t (q)\) for \(F (t, q)\).
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle F_t (F_s (q)) = F_{t + s} (q)\)</td>
        <td align="right">(2.10)</td>
      </tr>
    </table>
    <p>
      The map \(F\) in (2.9) is called a local flow genrated by the vector
      field \(X\). For each point \(q \in U\), the function \(F_t (q)\) is
      called a flow line of the local flow. Each flow line is an integral
      curve of \(X\). If a local flow \(F\) is defined on \(\mathbb{R} \times
      M\), then it is called a global flow. A vector field haing a global flow
      is called a <i>complete vector field</i>. If \(F\) is a global flow,
      then for every \(t \in \mathbb{R}\),
    </p>
    <center>
      \(\displaystyle F_t \circ F_{- t} = F_{- t} \circ F_t = F_0 =
      \mathbb{1}_M,\)
    </center>
    <p>
      so \(F_t : M \rightarrow M\) is a diffeomorphism. Thus, a global flow on
      \(M\) gives rise to a one-parameter group of diffeomorphisms of \(M\).
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">2.9</class>. </strong><i>A local
      flow about a point \(p\) in an open set \(U\) of a manifold is a
      \(C^{\infty}\) function</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle F :] - \varepsilon, \varepsilon [\times W \rightarrow
        U,\)
      </center></i>
    </p>
    <p>
      <i>where \(\varepsilon\) is a positive real numberand \(W\) is a
      neighborhood of \(p\) in \(U\), such that writing \(F_t (q) = F (t,
      q)\), we have </i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            \(F_0 (q) = q\) for all \(q \in W\),
          </p>
        </li>
        <li>
          <p>
            \(F_t (F_s (q)) = F_{t + s} (q)\) whenever both sides are defined.
          </p>
        </li>
      </ol></i>
    </div>
    <p>
      If \(F (t, q)\) is a local flow of the vector field \(X\) on \(U\), then
    </p>
    <center>
      \(\displaystyle F (0, q) = q \quad \operatorname{and} \quad
      \frac{\partial F}{\partial t} (0,
q) = X_{F (0, q)} = X_q .\)
    </center>
    <p>
      Thus, one can recover the vector field from its flow.
    </p>
    <p style="margin-top: 1em">
      <strong>例. </strong>The function \(F : \mathbb{R} \times
      \mathbb{R}^2 \rightarrow \mathbb{R}^2,\)
    </p>
    <center>
      \(\displaystyle F \left( t, \left[ \begin{array}{c}
  x\\
 
      y
\end{array} \right] \right) = \left[ \begin{array}{cc}
  \cos t & -
      \sin t\\
  \sin t & \cos t
\end{array} \right] \left[ \begin{array}{c}
 
      x\\
  y
\end{array} \right],\)
    </center>
    <p>
      is the global flow on \(\mathbb{R}^2\) genrated by the vector field
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle \begin{array}{rl}
  X_{(x, y)} & = \frac{\partial
        F}{\partial t} (t, (x, y)) |_{t = 0} = \left[
  \begin{array}{cc}
   
        - \sin t & - \cos t\\
    \cos t & - \sin t
  \end{array} \right]
        \left[ \begin{array}{c}
    x\\
    y
  \end{array} \right] |_{t =
        0}\\
  & = \left[ \begin{array}{cc}
    0 & - 1\\
    1 & 0
 
        \end{array} \right] \left[ \begin{array}{c}
    x\\
    y
 
        \end{array} \right] = \left[ \begin{array}{c}
    - y\\
    x
 
        \end{array} \right] = - y \frac{\partial}{\partial x} + x
 
        \frac{\partial}{\partial y} .
\end{array}\)
      </center>
    </p>
    <h3 id="auto-9">2.4<span style="margin-left: 1em"></span>The Lie Bracket<span style="margin-left: 1em"></span></h3>
    <p>
      Suppose \(X\) and \(Y\) are smooth vector fields on an open subset \(U\)
      of a manifold \(M\). We view \(X\) and \(Y\) as derivations on
      \(C^{\infty} (U)\). \(X Y\) does not satisfy the derivation property:if
      \(f, g \in C^{\infty} (U)\), then
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  X Y (f g) & = X ((Y f) g + f Y g)\\

      & = (X Y f) g + (Y f) (X g) + (X f) (Y g) + f (X Y g) .

\end{array}\)
</center>
<p>
We see that the two extra terms \((Y f) (X g)\) and \((X f) (Y g)\) that
make \(X Y\) not a derivation are symmetric in \(X\) and \(Y\). Thus, if
we compute \(Y X (f g)\) as well and subtract it from \(X Y (f g)\), the
extra terms will disappear, and \(X Y - Y X\) will be derivation of
\(C^{\infty} (U)\).
</p>
<p>
We define the Lie bracket \([X, Y]\) at \(p\) to be
</p>
<center>
\(\displaystyle [X, Y]_p f = (X_p Y - Y_p X) f\)
</center>
<p>
for any germ \(f\) of a \(C^{\infty}\) function at \(p\).
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.10</class>. </strong><i>If
\(X\) and \(Y\) are smooth vector fields on \(M\), then the vector field
\([X, Y]\) is also smooth on \(M\).</i>
</p>
<center>
\(\displaystyle [Y, X] = - [X, Y]\)
</center>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">2.11</class>. </strong>(Jacobian
identity). Check the Jacobian identity:
</p>
<center>
\(\displaystyle \sum_{\operatorname{cyclic}} [X, [Y, Z]] = 0.\)
</center>
<p style="margin-bottom: 1em">
<center>
\(\displaystyle \sum*{\operatorname{cyclic}} [X, [Y, Z]] = [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X,
Y]] .\)
</center>
</p>
<p style="margin-top: 1em">
<strong>定义 <class style="font-style: normal">2.12</class>. </strong><i>Let
\(K\) be a field. A Lie algebra over \(K\) is a vector space \(V\) over
\(K\) together with a product \([,] : V \times V \rightarrow V\), called
the bracket, satisfying the following properties: for all \(a, b \in K\)
and \(X, Y, Z \in V\),</i>
</p>
<div style="margin-bottom: 1em">
<i><ol>
<li>
<p>
(bilinearity) \([a X +\operatorname{bY}, Z] = a [X, Z] + b [Y,
Z],\)
</p>
<p>
\([Z, a X + b Y] = a [Z, X] + b [Z, Y]\),
</p>
</li>
<li>
<p>
(anticommutativity) \([Y, X] = - [X, Y]\),
</p>
</li>
<li>
<p>
(Jacobi identity) \(\sum*{\operatorname{cyclic}} [X, [Y, Z]] =
0\).
</p>
</li>
</ol></i>
</div>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>例. </strong>On any vector space \(V\), define \([X, Y] =
0\) for all \(X, Y \in V\). With this bracket, \(V\) becomes a Lie
algebra, called an <i>abelian Lie algebra</i>.
</p>
<p>
An abelian Lie algebra is trivially associative, but in general that
bracket of a Lie algebra need not be associative. So despite its name, a
Lie algebra is in genral not an algebra.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>例. </strong>If \(M\) is a manifold, then the vector space
\(\mathfrak{X} (M)\) of \(C^{\infty}\) vector fields on \(M\) is a real
Lie algebra with the Lie bracket \([,]\) as the bracket.
</p>
<p style="margin-top: 1em">
<strong>例. </strong>Let \(K^{n \times n}\) be the vector space of
all \(n \times n\) matrices over a field \(K\). Define for \(X, Y \in
K^{n \times n}\),
</p>
<center>
\(\displaystyle [X, Y] = X Y - Y X,\)
</center>
<p>
where \(X Y\) is the matrix product of \(X\) and \(Y\). With this
bracket, \(K^{n \times n}\) becomes a Lie algebra. The bilinearity and
anticommutativity of \([,]\) are immediate, while the Jacobi identity
follows from the same computation as in Exercise.
</p>
<p>
More generally, if \(A\) is any algebra over a field \(K\), then the
product
</p>
<center>
\(\displaystyle [x, y] = x y - y x, \quad x, y \in A,\)
</center>
<p style="margin-bottom: 1em">
makes \(A\) into a Lie algebra over \(K\).
</p>
<p style="margin-top: 1em">
<strong>定义 <class style="font-style: normal">2.13</class>. </strong><i>A
derivation of a Lie algebra \(V\) over a field \(K\) is a \(K\)-linear
map \(D : V \rightarrow V\) satisfying the product rule</i>
</p>
<p style="margin-bottom: 1em">
<i><center>
\(\displaystyle D [Y, Z] = [D Y, Z] + [Y, D Z] \quad
\operatorname{for}Y, Z \in V\)
</center></i>
</p>
<p style="margin-top: 1em">
<strong>例. </strong>Let \(V\) be a Lie algebra over a field
\(K\). For each \(X\) in \(V\), define \(\operatorname{ad}_X : V
\rightarrow V\) by
</p>
<center>
\(\displaystyle \operatorname{ad}\_X (Y) = [X, Y] .\)
</center>
<p>
We may rewrite the Jacobi identity in the form
</p>
<center>
\(\displaystyle [X, [Y, Z]] = [[X, Y], Z] + [Y, [X, Z]]\)
</center>
<p>
or
</p>
<center>
\(\displaystyle \operatorname{ad}\_X [Y, Z] = [\operatorname{ad}_X Y, Z] + [Y,
\operatorname{ad}_X Z],\)
</center>
<p style="margin-bottom: 1em">
which shows that \(\operatorname{ad}\_X : V \rightarrow V\) is a
derivation of \(V\).
</p>
<h3 id="auto-10">2.5<span style="margin-left: 1em"></span>The Pushforward of Vector Fields<span
    style="margin-left: 1em"></span></h3>
<p>
Let \(F : N \rightarrow M\) be a smooth map of manifolds and let
\(F_{\ast} : T*p N \rightarrow T*{F (p)} M\) be its differential at a
point \(p\) in \(N\). If \(X*p \in T_p N\), we call \(F*{\ast} (X*p)\)
the pushforward of the vector \(X_p\) at \(p\). This notion does not
extend in general to vector fields, since if \(X\) is a vector field on
\(N\) and \(z = F (p) = F (q)\) for two distinct points \(p, q \in N\),
then \(X_p\) and \(X_q\) are both push forward to tangent vectors at \(z
\in M\), but there is no reason why \(F*{\ast} (X*p)\) and \(F*{\ast}
(X*q)\) should be equal.
</p>
<p>
When \(F : N \rightarrow M\) is a diffeomorphism, \((F*{\ast} X)_{F (p)}
= F_{\ast, p} (X*p)\), \(F*{\ast} X\) is defined everywhere on \(M\).
</p>
<h3 id="auto-11">2.6<span style="margin-left: 1em"></span>Related Vector Fields<span style="margin-left: 1em"></span></h3>
<p style="margin-top: 1em">
<strong>定义 <class style="font-style: normal">2.14</class>. </strong><i>Let \(F
: N \rightarrow M\) be a smooth map of manifolds. A vector field \(X\)
on \(N\) is \(F\)-related to a vector field \(\bar{X}\) on \(M\) if for
all \(p \in N\),</i>
</p>
<div style="margin-bottom: 1em">
<i><table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle F*{\ast, p} (X_p) = \bar{X}*{F
(p)} .\)</td>
<td align="right">(2.11)</td>
</tr>
</table></i>
</div>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>例 <class style="font-style: normal">2.15</class>. </strong>(Pushforward by a
diffeomorphism). If \(F : N \rightarrow M\) is a diffeomorphism and
\(X\) is a vector field on \(N\), then the pushforward \(F*{\ast} X\) is
defined. By definition, the vector field \(X\) on \(N\) is \(F\)-related
to the vector field \(F*{\ast} X\) on \(M\). In subsection later, we
will see examples of vector fields related by a map \(F\) that is not a
diffeomorphism.
</p>
<p style="margin-top: 1em">
<strong>命题 <class style="font-style: normal">2.16</class>. </strong><i>Let \(F
: N \rightarrow M\) be a smooth map of manifolds. A vector field \(X\)
on \(N\) and a vector field X<sup class="wide">&macr;</sup> on \(M\) are
\(F\)-related if and only if for all \(g \in C^{\infty} (M)\),</i>
</p>
<p style="margin-bottom: 1em">
<i><center>
\(\displaystyle X (g \circ F) = (\bar{X} g) \circ F.\)
</center></i>
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.17</class>. </strong><i>Let \(F
: N \rightarrow M\) be a smooth map of manifolds. If the \(C^{\infty}\)
vector fields \(X\) and \(Y\) on \(N\) are \(F\)-related to the
\(C^{\infty}\) vector fields \(\bar{X}\) and Y<sup class="wide">&macr;</sup>,
respectively, on \(M\), then the Lie bracket \([X, Y]\) on \(N\) is
\(F\)-related to the Lie bracket \([\bar{X}, \bar{Y}]\) on \(M\).</i>
</p>
<h1 id="auto-12">1<span style="margin-left: 1em"></span>Lie Groups and Lie Algebra<span style="margin-left: 1em"></span></h1>
<p>
A Lie group is a <b>manifold</b> that is also a <b>group</b> such that
the group operations are <b>smooth</b>. Classical groups such as the
general and special linear groups over \(\mathbb{R}\) and over
\(\mathbb{C}\), orthogonal groups, unitary groups, and symplectic groups
are all Lie groups.
</p>
<p>
李群是个带有群结构的光滑流形，感觉叫李流形更好，李表示群结构。不过本来Lie可能是从对称性出发开始研究的，后来人们在定义的时候才加上流形的概念。所以另一个角度来说，李群是带有流形连续结构的群。一般和特殊线性群、正交群、酉群、辛群，终于知道这都是什么了。
</p>
<p>
A Lie group is a homogeneous space in the sense that left translation by
a group element \(g\) is a diffeomorphism of the group onto itself that
maps the identity elemnt to \(g\). Therefore, locally the group looks
the same around any point. To study the local structure of a Lie group,
it is enough to examine a neighborhood of the identity element. It is
not surprising that the tangent space at the identity of a Lie group
should play a key role.
</p>
<p>
The tangent space at the identity of a Lie group \(G\) turns out to have
a canonical bracket operation \([,]\) that makes it into a Lie algebra.
The tangent space \(T*e G\) with the bracket is called the Lie algebra
of the Lie group \(G\). The Lie algebra of a Lie group encodes within it
much information about the group.
向量空间加一个括号积得到一个李代数。
</p>
<p>
Lie's original motivation was to study the group of transformations of a
space as a continuous analogue of the group of permutations of a finite
set. Indeed, a diffeomorphism of a manifold \(M\) can be viewed as a
permutation of the points of \(M\). The interplay of group theory,
topology, and Linear algebra makes the theory of Lie groups and Lie
algebras a particularly rich and vibrant branch of mathematics. In this
chapter, we can but scratch the surface of this vast creation. For us,
Lie groups serve mainly as an important class of manifolds, and Lie
algebra as examples of tangent spaces.
</p>
<h2 id="auto-13">1<span style="margin-left: 1em"></span>Lie Groups<span style="margin-left: 1em"></span></h2>
<p>
Begin with some examples of matrix groups. The goal is to exhibit a
variety of methods for showing that a group is a Lie group and for
computing the dimension of a Lie group. A powerful tool, which we state
but do not prove, is the closed subgroup theorem. An abstract subgroup
that is a closed subset of a Lie group is itself a Lie group.
</p>
<p>
The matrix exponential gives rise to curves in a matrix group with a
given intitial vector. It is useful in computing the differential of a
map on a matrix group. As an example, we compute the differential of the
determinant map on the general linear group over \(\mathbb{R}\).
</p>
<h3 id="auto-14">1.1<span style="margin-left: 1em"></span>Examples of Lie Groups<span style="margin-left: 1em"></span></h3>
<p style="margin-top: 1em">
<strong>定义 <class style="font-style: normal">1.1</class>. </strong><i>A Lie
group is a \(C^{\infty}\) manifold \(G\) that is also a group such that
the two group operations, multiplication</i>
</p>
<p>
<i><center>
\(\displaystyle \mu : G \times G \rightarrow G, \quad \mu (a, b) = a
b\)
</center></i>
</p>
<p>
<i>and inverse</i>
</p>
<p>
<i><center>
\(\displaystyle \iota : G \rightarrow G, \quad \iota (a) = a^{- 1},\)
</center></i>
</p>
<p style="margin-bottom: 1em">
<i>are \(C^{\infty}\).</i>
</p>
<p>
For \(a \in G\), denote by \(\ell_a : G \rightarrow G\), \(\ell_a (x) =
\mu (a, x) = a x\), the operation of left multiplication by \(a\), and
by \(r_a : G \rightarrow G\), \(r_a (x) = x a\), the operation of right
multiplication by \(a\). We also call left and right multiplications
left and right translations.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>例 <class style="font-style: normal">1.2</class>. </strong>(Left
multiplication). For an element \(a\) in a Lie group \(G\), prove that
the left multiplication \(\ell_a : G \rightarrow G\) is a
diffeomorphism.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>定义 <class style="font-style: normal">1.3</class>. </strong><i>A map
\(F : H \rightarrow G\) between two Lie groups \(H\) and \(G\) is a Lie
group homomorphism if it is a \(C^{\infty}\) map and a group
homomorphism. </i>
</p>
<p>
The group homomorphism condition means that for all \(h, x \in H\),
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle F (h x) = F (h) F (x) .\)</td>
<td align="right">(1.1)</td>
</tr>
</table>
<p>
This may be rewritten in functional notation as
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle F \circ \ell_h = \ell*{F (h)} \circ
F \quad
\operatorname{for}\operatorname{all}h \in H.\)</td>
<td align="right">(1.2)</td>
</tr>
</table>
<p>
Let \(e*H, e_G\) be the identity elements of \(H\) and \(G\),
respectively. Taking \(h\) and \(x\) in (1.1) to be the identity
\(e_H\), it follows that \(F (e_H) = e_G\).
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">1.4</class>. </strong>(General linear
group). We showed that the general linear group
</p>
<center>
\(\displaystyle \operatorname{GL} (n, \mathbb{R}) = \{ A \in
\mathbb{R}^{n \times n} | \det A
\neq 0 \}\)
</center>
<p style="margin-bottom: 1em">
is a Lie group.
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">1.5</class>. </strong>(Special linear
group). The special linear group \(\operatorname{SL} (n, \mathbb{R})\)
is the subgroup of \(\operatorname{GL} (n, \mathbb{R})\) consisting of
matrices of determinant 1. \(\operatorname{SL} (n, \mathbb{R})\) is a
regular submanifold of dimension \(n^2 - 1\) of \(\operatorname{GL} (n,
\mathbb{R})\). The multiplication map
</p>
<center>
\(\displaystyle \bar{\mu} : \operatorname{SL} (n, \mathbb{R}) \times
\operatorname{SL} (n,
\mathbb{R}) \rightarrow \operatorname{SL} (n,
\mathbb{R})\)
</center>
<p>
is \(C^{\infty}\).
</p>
<p>
To see that the inverse map
</p>
<center>
\(\displaystyle \bar{\iota} : \operatorname{SL} (n, \mathbb{R})
\rightarrow \operatorname{SL}
(n, \mathbb{R})\)
</center>
<p>
is \(C^{\infty}\), let \(i : \operatorname{SL} (n, \mathbb{R})
\rightarrow \operatorname{GL} (n,
\mathbb{R})\) be the inclusion map and
\(\iota : \operatorname{GL} (n, \mathbb{R}) \rightarrow
\operatorname{GL} (n,
\mathbb{R})\) the inverse map of
\(\operatorname{GL} (n, \mathbb{R})\). As the composite of two
\(C^{\infty}\) maps,
</p>
<center>
\(\displaystyle \iota \circ i : \operatorname{SL} (n, \mathbb{R})
\xrightarrow{i}
\operatorname{GL} (n, \mathbb{R}) \xrightarrow{\iota}
\operatorname{GL} (n,
\mathbb{R})\)
</center>
<p style="margin-bottom: 1em">
is a \(C^{\infty}\) map. Since its image is contained in the regular
submanifold \(\operatorname{SL} (n, \mathbb{R})\), the induced map
\(\bar{\iota} : \operatorname{SL} (n, \mathbb{R}) \rightarrow
\operatorname{SL}
(n, \mathbb{R})\) is \(C^{\infty}\) by Theorem of
smooth maps into a submanifold. Thus, \(\operatorname{SL} (n,
\mathbb{R})\) is a Lie group.
</p>
<p>
An entirely similar argument proves that the complex special linear
group \(\operatorname{SL} (n, \mathbb{C})\) is also a Lie group.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>例 <class style="font-style: normal">1.6</class>. </strong>(Orthogonal group).
\(O (n)\) is the subgroup of \(\operatorname{GL} (n, \mathbb{R})\)
consisting of all matrices \(A\) satisfying \(A^T A = I\). Thus, \(O
(n)\) is the inverse image of \(I\) under the map \(f (A) = A^T A.\)
</p>
<p>
We showed that \(f : \operatorname{GL} (n, \mathbb{R}) \rightarrow
\operatorname{GL} (n,
\mathbb{R})\) has constant rank. By the
constant-rank level set theorem, \(O (n)\) is a regular submanifold of
\(\operatorname{GL} (n, \mathbb{R})\).
</p>
<p>
We determine the dimension of \(O (n)\) at the same time. We must first
redefine the target space of \(f\). Since \(A^T A\) is a symmetric
matrix, the image of \(f\) lies in \(S_n\), the vector space of all \(n
\times n\) real symmetric matrices.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>例 <class style="font-style: normal">1.7</class>. </strong>(Space of symmetric
matrices). Show that the vector space \(S_n\) of \(n \times n\) real
symmetric matrices has dimension \((n^2 + n) / 2\).
</p>
<p>
Consider the map \(f : \operatorname{GL} (n, \mathbb{R}) \rightarrow
S_n, f (A) = A^T A\). The tangent space of \(S_n\) at any point is
canonically isomorphic to \(S_n\) itself, because \(S_n\) is a vector
space. Thus, the image of the differential
</p>
<center>
\(\displaystyle f*{\ast, A} : T*A (\operatorname{GL} (n, \mathbb{R}))
\rightarrow T*{f (A)}
(S*n) \simeq S_n\)
</center>
<p>
lies in \(S_n\). For the differential \(f*{\ast, A}\) to be surjective,
the target space of \(f\) should be as small as possible.
</p>
<p>
We compute explicitly the differential \(f*{\ast, A}\). Since
\(\operatorname{GL} (n, \mathbb{R})\) is an open subset of
\(\mathbb{R}^{n \times n}\), its tangent space at any \(A \in
\operatorname{GL} (n, \mathbb{R})\) is
</p>
<center>
\(\displaystyle T_A (\operatorname{GL} (n, \mathbb{R})) = T_A
(\mathbb{R}^{n \times n})
=\mathbb{R}^{n \times n} .\)
</center>
<p>
For any matrix \(X \in \mathbb{R}^{n \times n}\), there is a curve \(c
(t)\) in \(\operatorname{GL} (n, \mathbb{R})\) with \(c (0) = A\) and
\(c' (0) = X\). By proposition before,
</p>
<center>
\(\displaystyle \begin{array}{rl}
f*{\ast, A} (X) & = \frac{d}{d x} f
(c (t)) |_{t = 0}\\
& = \frac{d}{d t} c (t)^T c (t) |_{t = 0}\\
& =
(c' (t)^T c (t) + c (t)^T c' (t)) |_{t = 0}\\
& = X^T A + A^T
X.
\end{array}\)
</center>
<p>
The surjectivity of \(f_{\ast, A}\) becomes the following question: if
\(A \in O (n)\) and \(B\) is any symmetric matrix in \(S*n\), does there
exist an \(n \times n\) matrix \(X\) such that
</p>
<center>
\(\displaystyle X^T A + A^T X = B ?\)
</center>
<p>
Note that since \((X^T A)^T = A^T X\), it is enough to solve
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle A^T X = \frac{1}{2} B,\)</td>
<td align="right">(1.3)</td>
</tr>
</table>
<p>
for then
</p>
<center>
\(\displaystyle X^T A + A^T X = \frac{1}{2} B^T + \frac{1}{2} B = B.\)
</center>
<p>
Equation (1.3) clearly has a solution: \(X = \frac{1}{2} (A^T)^{- 1}
B\). So \(f*{\ast, A} : T_A (\operatorname{GL} (n, \mathbb{R}))
\rightarrow S_n\) is surjective for all \(A \in O (n)\), and \(O (n)\)
is a regular level set of \(f\). By the regular level set theorem, \(O
(n)\) is a regular submanifold of \(\operatorname{GL} (n, \mathbb{R})\)
of dimension
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle \dim O (n) = n^2 - \dim S_n = n^2 -
\frac{n^2 + n}{2} = \frac{n^2 - n}{2} .\)</td>
<td align="right">(1.4)</td>
</tr>
</table>
<p>

    </p>
    <h3 id="auto-15">1.2<span style="margin-left: 1em"></span>Lie Subgroups<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">1.8</class>. </strong><i>A Lie
      subgroup of a Lie group \(G\) is (i) an abstract subgroup \(H\) that is
      (ii) an immersed submanifold via the inclusion map such that (iii) the
      group operation on \(H\) are \(C^{\infty}\).</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.9</class>. </strong>(Lines with
      irrational slope in a torus). Let \(G\) be the torus \(\mathbb{R}^2
      /\mathbb{Z}^2\) and \(L\) a line through the origin in \(\mathbb{R}^2\).
      The torus can also be represented by the unit square with the opposite
      edges identified. The image \(H\) of \(L\) under the projection \(\pi :
      \mathbb{R}^2 \rightarrow \mathbb{R}^2 /\mathbb{Z}^2\) is a closed curve
      if and only if the line \(L\) goes through another lattice point, say
      \((m, n) \in \mathbb{Z}^2\). This is the case if and only if the slope
      of \(L\) is \(n / m\), a rational number or \(\infty\); then \(H\) is
      the image of finitely many line segments on the unit square. It is a
      closed curve diffeomorphic to a circle and is a regular sublmanifold of
      \(\mathbb{R}^2 /\mathbb{Z}^2\).
    </p>
    <p>
      If the slope of \(L\) is irrational, then its image \(H\) on the torus
      will never close up. In this case the restriction to \(L\) of the
      projection map, \(f = \pi |_L : L \rightarrow \mathbb{R}^2
      /\mathbb{Z}^2\), is a one-to-one immersion. We give \(H\) the topology
      and manifold structure induced from \(f\). It can be shown that \(H\) is
      a dense subset of the torus. Thus, \(H\) is an immersed submanifold but
      not a regular submanifold of the torus \(\mathbb{R}^2 /\mathbb{Z}^2\).
    </p>
    <p>
      Whatever the slope of \(L\), its image \(H\) in \(\mathbb{R}^2
      /\mathbb{Z}^2\) is an abstract subgroup of the torus, an immersed
      submanifold, and a Lie group. Therefore, \(H\) is a Lie subgroup of the
      torus.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.10</class>. </strong>(Induced topology
      versus subspace topology). Suppose \(H \subset \mathbb{R}^2
      /\mathbb{Z}^2\) is the image of a line \(L\) with irrational slope in
      \(\mathbb{R}^2\). We call the topology on \(H\) induced from the
      bijection \(f : L \xrightarrow{\sim} H\) the induced topology and the
      topology on \(H\) as a subset of \(\mathbb{R}^2 /\mathbb{Z}^2\) the
      subspace topology. Compare these two topologies: is one a subset of the
      other?
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1.11</class>. </strong><i>If
      \(H\) is an abstract subgroup and a regular sublmanifold of a Lie group
      \(G\), then it is a Lie subgroup of \(G\).</i>
    </p>
    <p>
      A subgroup in Proposition 1.11 is called an embedded Lie subgroup,
      because the inclusion map \(i : H \rightarrow G\) of a regular
      submanifold is an embedding.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>We showed that the subgroups
      \(\operatorname{SL} (n, \mathbb{R})\) and \(O (n)\) of
      \(\operatorname{GL} (n, \mathbb{R})\) are both regular submanifolds.
      They are embedded Lie subgroups.
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">1.12</class>. </strong><i>(Closed
      subgroup theorem). A closed subgroup of a Lie group is an embedded Lie
      subgroup.</i>
    </p>
    <h3 id="auto-16">1.3<span style="margin-left: 1em"></span>The Matrix Exponential<span style="margin-left: 1em"></span></h3>
    <p>
      A norm on a vector space \(V\) is a real-valued function \(\| \cdot \| :
      V \rightarrow \mathbb{R}\) satisfying the following three properties:
      for all \(r \in \mathbb{R}\) and \(v, w \in V\),
    </p>
    <ol>
      <li>
        <p>
          (positive-definiteness) \(\| v \| \geq 0\) with equality if and only
          if \(v = 0\),
        </p>
      </li>
      <li>
        <p>
          (positive homogeneity) \(\| r v \| = | r |  \| v \|\),
        </p>
      </li>
      <li>
        <p>
          (subadditivity) \(\| v + w \| \leq \| v \| + \| w \|\).
        </p>
      </li>
    </ol>
    <p>
      A vector space with a norm is called a normed vector space. The vector
      space \(\mathbb{R}^{n \times n} \simeq \mathbb{R}^{n^2}\) of all \(n
      \times n\) real matrices can be given the Euclidean norm: for \(X =
      [x_{i j}] \in \mathbb{R}^{n \times n}\)
    </p>
    <center>
      \(\displaystyle \| X \| = \left( \sum x_{i j}^2 \right)^{1 / 2} .\)
    </center>
    <p>
      The matrix exponential \(e^X\) of a matrix \(X \in \mathbb{R}^{n \times
      n}\) is defined by the same formula as the exponential of a real number
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle e^X = I + X + \frac{1}{2!} X^2 +
        \frac{1}{3!} X^3 + \cdots,\)</td>
        <td align="right">(1.5)</td>
      </tr>
    </table>
    <p>
      where \(I\) is the \(n \times n\) identity matrix.
    </p>
    <p>
      A normed algebra \(V\) is a normed vector space that is also an algebra
      over \(\mathbb{R}\) satisfying the submultiplicative property: for all
      \(v, w \in V\), \(\| v w \| \leqslant \| v \|  \| w \|\).
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1.13</class>. </strong><i>For
      \(X, Y \in \mathbb{R}^{n \times n}\), \(\| X Y \| \leq \| X \|  \| Y
      \|\).</i>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.14</class>. </strong><i>Let
      \(V\) be a normed algebra.</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            if \(a \in V\) and \(s_m\) is a sequencce in \(V\) that converges
            to \(s\), then \(a s_m\) converges to \(a s\).
          </p>
        </li>
        <li>
          <p>
            if \(a \in V\) and \(\sum_{k = 0}^{\infty} b_k\) is a convergent
            series in \(V\), then \(a \sum_k b_k = \sum_k a b_k\).
          </p>
        </li>
      </ol></i>
    </div>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例 <class style="font-style: normal">1.15</class>. </strong>(Distributivity
      over a convergent series).
    </p>
    <p>
      Unlike the exponential of real numbers, when \(A\) and \(B\) are \(n
      \times n\) matrices with \(n > 1\), it is not necessarily true that
    </p>
    <center>
      \(\displaystyle e^{A + B} = e^A e^B .\)
    </center>
    <p style="margin-top: 1em">
      <strong>例 <class style="font-style: normal">1.16</class>. </strong>(Exponential of
      commuting matrices). Prove that if \(A\) and \(B\) are commuting \(n
      \times n\) matrices, then
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle e^A e^B = e^{A + B} .\)
      </center>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">1.17</class>. </strong><i>For \(X
      \in \mathbb{R}^{n \times n}\),</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \frac{d}{d t} e^{t X} = X e^{t X} = e^{t X} X.\)
      </center></i>
    </p>
    <h3 id="auto-17">1.4<span style="margin-left: 1em"></span>The Trace of a Matrix<span style="margin-left: 1em"></span></h3>
    <p>
      Define the trace of an \(n \times n\) matrix to be the sum of its
      diagonal entries:
    </p>
    <center>
      \(\displaystyle \operatorname{tr} (X) = \sum_{i = 1}^n x_{i i} .\)
    </center>
    <p style="margin-top: 1em">
      <strong>引理 <class style="font-style: normal">1.18</class>. </strong><i></i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            For any two matrices \(X, Y \in \mathbb{R}^{n \times n}\),
            \(\operatorname{tr} (X Y) =\operatorname{tr} (Y X)\).
          </p>
        </li>
        <li>
          <p>
            For \(X \in \mathbb{R}^{n \times n}\) and \(A \in
            \operatorname{GL} (n, \mathbb{R})\), \(\operatorname{tr} (A X A^{-
            1}) =\operatorname{tr} (X)\).
          </p>
        </li>
      </ol></i>
    </div>
    <p>
      The eigenvalues of an \(n \times n\) matrix \(X\) are the roots of the
      polynomial equation \(\det (I - \lambda X) = 0\). Over the field of
      complex numbers, which is algebraically closed, such an equation
      necessarily has \(n\) roots, counted with multiplicity. Thus, the
      advantage of allowing complex numbers is that every \(n \times n\)
      matrix, real or complex, has \(n\) complex eigenvalues, counted with
      multiplicity, whereas a real matrix need not have any real eigenvalue.
    </p>
    <center>
      \(\displaystyle \det (\lambda I - A X A^{- 1}) = \det (A (\lambda I - X)
      A^{- 1}) = \det

(\lambda I - X) .\)
</center>
<center>
\(\displaystyle \det \left( \lambda I - \left[ \begin{array}{ccc}

      \lambda_1 &  & \ast\\

& \ddots & \\
0 & & \lambda*n
\end{array}
\right] \right) = \prod*{i = 1}^n (\lambda - \lambda*i) .\)
</center>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">1.19</class>. </strong><i>The
trace of a matrix, real or complex, is equal to the sum of its complex
eigenvalues.</i>
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">1.20</class>. </strong><i>For any
\(X \in \mathbb{R}^{n \times n}\), \(\det (e^X) =
e^{\operatorname{tr}X}\).</i>
</p>
<p>
This is one reason why the matrix exponential is so useful, for it
allows us to write down explicitly a curve in \(\operatorname{GL} (n,
\mathbb{R})\) with a given initial point and a given initial velocity.
</p>
<center>
\(\displaystyle c (0) = e^{0 X} = e^0 = I\)
</center>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle c' (0) = \frac{d}{d t} e^{t X} |*{t
= 0} = X e^{t X} |_{t = 0} = X.\)</td>
<td align="right">(1.6)</td>
</tr>
</table>
<h3 id="auto-18">1.5<span style="margin-left: 1em"></span>The Differential of \(\det\) at the
Identity<span style="margin-left: 1em"></span></h3>
<p>
Let \(\det : \operatorname{GL} (n, \mathbb{R}) \rightarrow \mathbb{R}\)
be the determinant map. The tangent space \(T_I \operatorname{GL} (n,
\mathbb{R})\) to \(\operatorname{GL} (n, \mathbb{R})\) at the identity
matrix \(I\) is the vector space \(\mathbb{R}^{n \times n}\) and the
tangent space \(T_1 \mathbb{R}\) to \(\mathbb{R}\) at 1 is
\(\mathbb{R}\). So
</p>
<center>
\(\displaystyle \det_{\ast, I} : \mathbb{R}^{n \times n} \rightarrow
\mathbb{R}.\)
</center>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">1.21</class>. </strong><i>For any
\(X \in \mathbb{R}^{n \times n}\), \(\det*{\ast, I} (X)
=\operatorname{tr}X\).</i>
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">1.22</class>. </strong>Special orthogonal
group \(\operatorname{SO} (2)\)
</p>
<p>
The special orthogonal group \(\operatorname{SO} (n)\) is defined to be
the subgroup of \(O (n)\) consisting of matrices of determinant 1. Show
that every matrix \(A \in \operatorname{SO} (2)\) can be written in the
form
</p>
<center>
\(\displaystyle A = \left[ \begin{array}{cc}
a & c\\
b &
d
\end{array} \right] = \left[ \begin{array}{cc}
\cos \theta & - \sin
\theta\\
\sin \theta & \cos \theta
\end{array} \right]\)
</center>
<p style="margin-bottom: 1em">
for some real number \(\theta\). Then prove that \(\operatorname{SO}
(2)\) is diffeomorphic to the circle \(S^1\).
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">1.23</class>. </strong>Unitary group
</p>
<p>
The unitary group \(U (n)\) is defined to be
</p>
<center>
\(\displaystyle U (n) = \{ A \in \operatorname{GL} (n, \mathbb{C}) |
\bar{A} A^T = I \},\)
</center>
<p style="margin-bottom: 1em">
where \(\bar{A} \) denotes the complex conjugate of \(A\), the matrix
obtained from \(A\) by conjugating every entry of \(A\): \((\bar{A})*{i
j} = \bar{a}\_{i j}\). Show that \(U (n)\) is a regular submanifold of
\(\operatorname{GL} (n, \mathbb{C})\) and that \(\dim U (n) = n^2\).
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">1.24</class>. </strong>Special unitary
group \(\operatorname{SU} (2)\)
</p>
<p>
The special unitary group \(\operatorname{SU} (n)\) is defined to be the
subgroup of \(U (n)\) consisting of matrices of determinant 1.
</p>
<ol style="margin-bottom: 1em">
<li>
<p>
Show that \(\operatorname{SU} (2)\) can also be described as the set

        </p>
        <center>
          \(\displaystyle \operatorname{SU} (2) = \left\{ \left[
          \begin{array}{cc}

a & - \bar{b}\\
b & \bar{a}
\end{array}
\right] \in \mathbb{C}^{2 \times 2} |a \bar{a} + b \bar{b} =
1
\right\} .\)
</center>
<p>
Hint: write out the condition \(A^{- 1} = \bar{A}^T\) in terms of
the entries.
</p>
</li>
<li>
<p>
Show that \(\operatorname{SU} (2)\) is diffeomorphic to the
three-dimensional sphere
</p>
<center>
\(\displaystyle S^3 = \{ (x_1, x_2, x_3, x_4) \in \mathbb{R}^4
|x_1^2 + x_2^2 + x_3^2 + x_4^2
= 1 \} .\)
</center>
</li>
</ol>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">1.25</class>. </strong>Complex symplectic
group
</p>
<p>
Let \(J\) be a \(2 n \times 2 n\) matrix
</p>
<center>
\(\displaystyle J = \left[ \begin{array}{cc}
0 & I_n\\

- I*n &
0
\end{array} \right],\)
</center>
<p>
where \(I_n\) denotes the \(n \times n\) identity matrix. The complex
symplectic group \(\operatorname{Sp} (2 n, \mathbb{C})\) is defined to
be
</p>
<center>
\(\displaystyle \operatorname{Sp} (2 n, \mathbb{C}) = \{ A \in
\operatorname{GL} (2 n,
\mathbb{C}) |A^T J A = J \} .\)
</center>
<p style="margin-bottom: 1em">
Show that \(\operatorname{Sp} (2 n, \mathbb{C})\) is a regular
submanifold of \(\operatorname{GL} (2 n, \mathbb{C})\) and compute its
dimension.
</p>
<h2 id="auto-19">2<span style="margin-left: 1em"></span>Lie Algebra<span style="margin-left: 1em"></span></h2>
<p>
In a Lie group \(G\), because left translation by an element \(g \in G\)
is a diffeomorphism that maps a neighborhood of the identity to a
neighborhood of \(g\), all the local information about the groupis
concentrated in a neighborhood of the identity, and the tangent space at
the identity assumes a special improtance.
</p>
<p>
Moreover, one can give the tangent space \(T_e G\) a Lie bracket
\([,]\), so that in addition to being a vector space, it becomes a Lie
algebra, called the Lie algebra of the Lie group. The goal of this
section is to define the Lie algebra structure on \(T_e G\) and to
identify the Lie algebras of a few classical groups.
</p>
<p>
The differential of a Lie group homomorphism becomes a Lie algebra
homomorphism. We thus obtain a functor from the category of Lie groups
and Lie group homomorphisms to the category of Lie algebras and Lie
algebra homomorphisms. This is the beginning of a rewarding program, to
understand the structure and representations of Lie groups through a
study of their Lie algebras.
</p>
<h3 id="auto-20">2.1<span style="margin-left: 1em"></span>Tangent Space at the Identity of a Lie
Group<span style="margin-left: 1em"></span></h3>
<p>
The diffeomorphism \(\ell_g\) takes the identity element \(e\) to the
element \(g\) and induces an isomorphism of tangent spaces
</p>
<center>
\(\displaystyle \ell*{g \ast} = (\ell*g)*{\ast, e} : T*e (G) \rightarrow
T_g (G) .\)
</center>
<p>
Thus, if we can describe the tangent space \(T_e (G)\) at the identity,
then \(\ell*{g \ast} T*e (G)\) will give a description of the tangent
space \(T_g (G)\) at any point \(g \in G\).
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">2.1</class>. </strong>(The tangent space
to \(\operatorname{GL} (n, \mathbb{R}) \operatorname{at}I\)).
</p>
<p style="margin-bottom: 1em">
We identified the tangent space of \(\operatorname{GL} (n, \mathbb{R})\)
at any point \(g \in \operatorname{GL} (n, \mathbb{R})\) as
\(\mathbb{R}^{n \times n}\), the vector space of all \(n \times n\) real
matrices.We also identified the isomorphism \(\ell*{g \ast} : T*I
(\operatorname{GL} (n, \mathbb{R})) \rightarrow T_g
(\operatorname{GL}
(n, \mathbb{R}))\) as left multiplication by \(g : X \mapsto g X\).
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">2.2</class>. </strong>(The tangent space
to \(\operatorname{SL} (g, \mathbb{R})\) at \(I\)). We begin by finding
a condition that a tangent vector \(X\) in \(T_I (\operatorname{SL} (n,
\mathbb{R}))\) must satisfy. By Proposition before, there is a curve \(c
:] - \varepsilon, \varepsilon [\rightarrow \operatorname{SL} (n,
\mathbb{R})\) with \(c (0) = I\) and \(c' (0) = X\). Being in
\(\operatorname{SL} (n, \mathbb{R})\), this curve satisfies
</p>
<center>
\(\displaystyle \det c (t) = 1\)
</center>
<p>
for all \(t\) in the domain \(] - \varepsilon, \varepsilon [\). We now
differentiate both sides with respect to \(t\) and evaluate at \(t =
0\). On the left-hand side, we have
</p>
<center>
\(\displaystyle \begin{array}{rl}
\frac{d}{d t} \det (c (t)) |*{t = 0}
& = (\det \circ c)_{\ast} \left(
\frac{d}{d t} |\_0 \right)\\
& =
\det_{\ast, I} \left( c*{\ast} \frac{d}{d t} |\_0 \right)\\
& =
\det*{\ast, I} (c' (0))\\
& = \det*{\ast, I} (X)\\
&
=\operatorname{tr} (X)
\end{array}\)
</center>
<p>
Thus,
</p>
<center>
\(\displaystyle \operatorname{tr} (X) = \frac{d}{d t} 1|*{t = 0} = 0.\)
</center>
<p>
So the tangent space \(T*I (\operatorname{SL} (n, \mathbb{R}))\) is
contained in the subspace \(V\) of \(\mathbb{R}^{n \times n}\) defined
by
</p>
<center>
\(\displaystyle V = \{ X \in \mathbb{R}^{n \times n} |\operatorname{tr}X
= 0 \} .\)
</center>
<p style="margin-bottom: 1em">
Since \(\dim V = n^2 - 1 = \dim T_I (\operatorname{SL} (n,
\mathbb{R}))\), the two spaces must be equal.
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.3</class>. </strong><i>The
tangent space \(T_I (\operatorname{SL} (n, \mathbb{R}))\) at the
identity of the special linear group \(\operatorname{SL} (n,
\mathbb{R})\) is the subspace of \(\mathbb{R}^{n \times n}\) consisting
of all \(n \times n\) matrices of trace 0.</i>
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">2.4</class>. </strong>(The tangent space
to \(O (n)\) at \(I\)). Choose a curve \(c (t)\) in \(O (n)\) defined on
a small interval containing 0 such that \(c (0) = I\) and \(c' (0) =
X\). Since \(c (t)\) is in \(O (n)\),
</p>
<center>
\(\displaystyle c^T (t) c (t) = I.\)
</center>
<p>
Differentiating both sides with respect to \(t\) using the matrix
product rule gives
</p>
<center>
\(\displaystyle c' (t)^T c (t) + c (t)^T c' (t) = 0.\)
</center>
<p>
Evaluating at \(t = 0\) gives
</p>
<center>
\(\displaystyle X^T + X = 0.\)
</center>
<p style="margin-bottom: 1em">
Thus, \(X\) is a skew-symmetric matrix.
</p>
<p>
Let \(K_n\) be the space of all \(n \times n\) real skew-symmetric
matrices.
</p>
<center>
\(\displaystyle \dim K_n = \frac{n^2
-\#\operatorname{diagonal}\operatorname{entries}}{2} =
\frac{1}{2} (n^2 - n) .\)
</center>
<p>
We have shown that
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle T_I (O (n)) \subset K_n .\)</td>
<td align="right">(2.1)</td>
</tr>
</table>
<center>
\(\displaystyle \dim T_I (O (n)) = \dim O (n) = \frac{n^2 - n}{2} .\)
</center>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.5</class>. </strong><i>The
tangent space \(T_I (O (n))\) of the orthogonal group \(O (n)\) at the
identity is the subspace of \(\mathbb{R}^{n \times n}\) consisting of
all \(n \times n\) skew-symmetric matrices.</i>
</p>
<h3 id="auto-21">2.2<span style="margin-left: 1em"></span>Left-Invariant Vector Fields on a Lie
Group<span style="margin-left: 1em"></span></h3>
<p>
A left-invariant vector field \(X\) is completely determined by its
value \(X_e\) at the identity, since
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle X_g = \ell*{g \ast} (X*e) .\)</td>
<td align="right">(2.2)</td>
</tr>
</table>
<p>
Conversely, given a tangent vector \(A \in T_e (G)\), we can define a
vector field \(\tilde{A}\) on \(G\) by (2.2):\((\tilde{A})\_g = \ell*{g
\ast} A\). So defined, the vector field \(\tilde{A}\) is left-invariant.
We call \(\tilde{A}\) the left-invariant vector field on \(G\) generated
by \(A \in T*e G\). Let \(L (G)\) be the vector space of all
left-invariant vector fields on \(G\). Then there is a one-to-one
correspondence
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle T_e (G) \leftrightarrow L (G)\)</td>
<td align="right">(2.3)</td>
</tr>
</table>
<center>
\(\displaystyle X_e \leftrightarrow X,\)
</center>
<center>
\(\displaystyle A \mapsto \tilde{A} .\)
</center>
<p>
The correspondence is in fact a vector space isomorphism.
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">2.6</class>. </strong>(Left-invariant
vector fields on \(\mathbb{R}\)).
</p>
<center>
\(\displaystyle \ell_g (x) = g + x.\)
</center>
<p>
Let us compute \(\ell*{g \ast} (d / d x|_0)\).
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle \ell_{g \ast} \left( \frac{d}{d x}
|_0 \right) = a \frac{d}{d x} |\_g .\)</td>
<td align="right">(2.4)</td>
</tr>
</table>
<p>
Apply both sides to function \(f (x) = x\), gives \(a = 1\).
</p>
<p>
Thus,
</p>
<p style="margin-bottom: 1em">
<center>
\(\displaystyle \ell_{g \ast} \left( \frac{d}{d x} |_0 \right) =
\frac{d}{d x} |\_g .\)
</center>
</p>
<p style="margin-top: 1em">
<strong>例 <class style="font-style: normal">2.7</class>. </strong>(Left-invariant
vector fields on \(\operatorname{GL} (n, \mathbb{R})\)).
</p>
<table style="margin-bottom: 1em" width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle \sum a_{i j}
\frac{\partial}{\partial x*{i j}} |\_g \leftrightarrow [a*{i j}]
.\)</td>
<td align="right">(2.5)</td>
</tr>
</table>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.8</class>. </strong><i>Any
left-invariant vector field \(X\) on a Lie group \(G\) is
\(C^{\infty}\).</i>
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.9</class>. </strong><i>If \(X\)
and \(Y\) are left-invariant vector fields on \(G\), then so is \([X,
Y]\).</i>
</p>
<h3 id="auto-22">2.3<span style="margin-left: 1em"></span>The Lie Algebra of a Lie Group<span style="margin-left: 1em"></span></h3>
<p>
Recall that a Lie algebra is a vector space \(\mathfrak{g}\) together
with a bracket, i.e., an anticommutative bilinear map \([,] :
\mathfrak{g} \times \mathfrak{g} \rightarrow \mathfrak{g}\) that
satisfies the Jacobi identity. A Lie subalgebra of a Lie algebra is a
vector subspace \(\mathfrak{h} \subset \mathfrak{g}\) that is closed
under the bracket \([,]\). The space \(L (G)\) of left-invariant vector
fields on a Lie group \(G\) is closed under the Lie bracket \([,]\) and
is therefore a Lie subalgebra of the Lie algebra \(\mathfrak{X} (G)\) of
all \(C^{\infty}\) vector fields on \(G\).
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle [A, B] = [\tilde{A}, \tilde{B}]_e
.\)</td>
<td align="right">(2.6)</td>
</tr>
</table>
<p style="margin-top: 1em">
<strong>命题 <class style="font-style: normal">2.10</class>. </strong><i>If \(A,
B \in T_e G\) and \(\tilde{A}, \tilde{B}\) are the left-invariant vector
fields they generate, then</i>
</p>
<p style="margin-bottom: 1em">
<i><center>
\(\displaystyle [\tilde{A}, \tilde{B}] = [A, B] \widetilde{} .\)
</center></i>
</p>
<h3 id="auto-23">2.4<span style="margin-left: 1em"></span>The Lie Bracket on \(\mathfrak{g
\mathfrak{l}} (n, \mathbb{R})\)<span style="margin-left: 1em"></span></h3>
<p>
We identified a tangent vector in \(T_I (\operatorname{GL} (n,
\mathbb{R}))\) with a matrix \(A \in \mathbb{R}^{n \times n}\) via
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle \sum a_{i j}
\frac{\partial}{\partial x*{i j}} |\_I \leftrightarrow [a*{i j}]
.\)</td>
<td align="right">(2.7)</td>
</tr>
</table>
<p>
The tangent space \(T*I \operatorname{GL} (n, \mathbb{R})\) with its Lie
algebra structure is denoted by \(\mathfrak{g \mathfrak{l}} (n,
\mathbb{R})\).
</p>
<p style="margin-top: 1em">
<strong>命题 <class style="font-style: normal">2.11</class>. </strong><i>Let
</i>
</p>
<p>
<i><center>
\(\displaystyle A = \sum a*{i j} \frac{\partial}{\partial x*{i j}}
\mathrel{|}\_I, \quad B =
\sum b*{i j} \frac{\partial}{\partial x*{i
j}} |\_I \in T_I (\operatorname{GL}
(n, \mathbb{R}))\)
</center></i>
</p>
<p>
<i>If </i>
</p>
<i><table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle [A, B] = [\tilde{A}, \tilde{B}]\_I =
\sum c*{i j} \frac{\partial}{\partial x*{i
j}} |\_I,\)</td>
<td align="right">(2.8)</td>
</tr>
</table></i>
<p>
<i>then</i>
</p>
<p>
<i><center>
\(\displaystyle c*{i j} = \sum*k a*{i k} b*{k j} - b*{i k} a*{k j} .\)
</center></i>
</p>
<p>
<i>Thus, if derivations are identified with matrices via (2.7), then
</i>
</p>
<p style="margin-bottom: 1em">
<i><center>
\(\displaystyle [A, B] = A B - B A.\)
</center></i>
</p>
<h3 id="auto-24">2.5<span style="margin-left: 1em"></span>The Pushforward of Left-Invariant Vector
Fields<span style="margin-left: 1em"></span></h3>
<p>
<center>
<script type="text/tikz" data-tex-packages='{ "amsmath": "", "amssymb": "", "amsfonts": "", "tikz-cd": "" }'>
\begin{tikzcd}
T_eH \arrow[r, "{F*{_,e}}"] \arrow[d, "\simeq"] & T*eG \arrow[d, "\simeq"] & {} & A \arrow[r, maps to] \arrow[d, maps to] & {F*{_,e}A} \arrow[d, maps to] \\
L(H) \arrow[r, dashed] & L(G) & {} & \tilde{A} \arrow[r, dashed, maps to] & {(F*{\*,e}A)\tilde{}}  
 \end{tikzcd}
</script>
</center>
</p>
<p style="margin-top: 1em">
<strong>定义 <class style="font-style: normal">2.12</class>. </strong><i>Let \(F
: H \rightarrow G\) be a Lie group homomorphism. Define \(F*{\ast} : L
(H) \rightarrow L (G)\) by</i>
</p>
<p>
<i><center>
\(\displaystyle F*{\ast} (\tilde{A}) = (F*{\ast, e} A) \widetilde{}\)
</center></i>
</p>
<p style="margin-bottom: 1em">
<i>for all \(A \in T*e H\).</i>
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>命题 <class style="font-style: normal">2.13</class>. </strong><i>If \(F
: H \rightarrow G\) is a Lie group homomorphism and \(X\) is a
left-invariant vector field on \(H\), then the left-invariant vector
field \(F*{\ast} X\) on \(G\) is \(F\)-related to the left-invariant
vector field \(X\).</i>
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle F*{\ast, h} (X_h) = (F*{\ast} X)_{F
(h)} .\)</td>
<td align="right">(2.9)</td>
</tr>
</table>
<h3 id="auto-25">2.6<span style="margin-left: 1em"></span>The Differential as a Lie Algebra
Homomorphism<span style="margin-left: 1em"></span></h3>
<p style="margin-top: 1em">
<strong>命题 <class style="font-style: normal">2.14</class>. </strong><i>If \(F
: H \rightarrow G\) is a Lie group homomorphism, then its differential
at the identity:</i>
</p>
<p>
<i><center>
\(\displaystyle F_{\ast} = F*{\ast, e} : T_e H \rightarrow T_e G,\)
</center></i>
</p>
<p>
<i>is a Lie algebra homomorphism, i.e., a linear map such that for all
\(A, B \in T_e H\),</i>
</p>
<p style="margin-bottom: 1em">
<i><center>
\(\displaystyle F*{\ast} [A, B] = [F_{\ast} A, F_{\ast} B] .\)
</center></i>
</p>
<p>
Suppose \(H\) is a Lie subgroup of a Lie group \(G\), with inclusion map
\(i : H \rightarrow G\). Since \(i\) is an immersion, its differential
</p>
<center>
\(\displaystyle i*{\ast} : T_e H \rightarrow T_e G\)
</center>
<p>
is injective.
</p>
<table width="100%">
<tr>
<td width="100%" align="center">\(\displaystyle i*{\ast} ([X, Y]_{T_e H}) =
[i_{\ast} X, i*{\ast} Y]*{T*e G} .\)</td>
<td align="right">(2.10)</td>
</tr>
</table>
<p>
This shows that if \(T_e H\) is identified with a subspace of \(T_e G\)
via \(i*{\ast}\), then the bracket on \(T_e H\) is the restriction of
the bracket on \(T_e G\) to \(T_e H\).
</p>
<p>
In general, the Lie algebra of the classical groups are denoted by
gothic letters. For example, the Lie algebra of \(\operatorname{GL} (n,
\mathbb{R})\), \(\operatorname{SL} (n, \mathbb{R})\), \(O (n)\), and \(U
(n)\) are denoted by \(\mathfrak{g \mathfrak{l}} (n, \mathbb{R})\),
\(\mathfrak{s \mathfrak{l}} (n, \mathbb{R})\), \(\mathfrak{o} (n)\), and
\(\mathfrak{u} (n)\), respectively. By (2.10) and proposition 2.11, the
Lie algebra structure on \(\mathfrak{s \mathfrak{l}} (n, \mathbb{R})\),
\(\mathfrak{o} (n)\), and \(\mathfrak{u} (n)\) are given by
</p>
<center>
\(\displaystyle [A, B] = A B - B A,\)
</center>
<p>
as on \(\mathfrak{g \mathfrak{l}} (n, \mathbb{R})\).
</p>
<p style="margin-top: 1em; margin-bottom: 1em">
<strong>注意 <class style="font-style: normal">2.15</class>. </strong>A
fundamental theorem in Lie group theory asserts the existence of a
one-to-one correspondence between the connected Lie subgroups of a Lie
group \(G\) and the Lie subalgebras of its Lie algebra \(\mathfrak{g}\).
For the torus \(\mathbb{R}^2 /\mathbb{Z}^2\), the Lie algebra
\(\mathfrak{g}\) has \(\mathbb{R}^2\) as the underlying vector space and
the one-dimensional Lie subalgebra are all the lines through the origin.
Each line through the origin in \(\mathbb{R}^2\) is a subgroup of
\(\mathbb{R}^2\) under addition. Its image under the quotient map
\(\mathbb{R}^2 \rightarrow \mathbb{R}^2 /\mathbb{Z}^2\) is a subgroup of
the torus \(\mathbb{R}^2 /\mathbb{Z}^2\). If a line has rational slope,
then its iamge is a regular submanifold of the torus. If a line has
irrational slope, then its image is onlyu an immersed submanifold of the
torus. According to correspondence theorem just quoted, the
one-dimensional connected Lie subgroups of the torus are the images of
all the lines through the origin. Note that if a Lie subgroup had been
defined as a subgroup that is also a regular submanifold, then one would
have to exclude all the lines with irrational slopes as Lie subgroups of
the torus, and it would not be possible to have a one-to-one
correspondence between the connected subgroups of a Lie group and the
Lie subalgebras of its Lie algebra. It is because of our desire for such
a correspondence that a Lie subgroup of a Lie group is defined to be a
subgroup that is also an immersed submanifold.
</p>
<div style="margin-top: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%"><strong>题目 <class style="font-style: normal">2.1</class>.
</strong>Skew-Hermitian matrices</font>
</p>
</div>
<div style="margin-bottom: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%">A complete matrix \(X \in \mathbb{C}^{n \times n}\) is
skew-Hermitian if its conjugate transpose \(\bar{X}^T\) is equal to
\(- X\). Let \(V\) be the vector spacce of \(n \times n\)
skew-Hermitian matrices. Show that \(\dim V = n^2\).</font>
</p>
</div>
<div style="margin-top: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%"><strong>题目 <class style="font-style: normal">2.2</class>.
</strong>Lie algebra of a unitary group</font>
</p>
</div>
<div style="margin-bottom: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%">Show that the tangent space at the identity \(I\) of the
unitary group \(U (n)\) is the vector space of \(n \times n\)
skew-Hermitian matrices.</font>
</p>
</div>
<div style="margin-top: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%"><strong>题目 <class style="font-style: normal">2.3</class>.
</strong>Lie algebra of a symplectic group</font>
</p>
</div>
<div style="margin-bottom: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%">Symplectic group \(\operatorname{Sp} (n)\). Show that the
tangent space at the identity \(I\) of the symplectic group
\(\operatorname{Sp} (n) \subset \operatorname{GL} (n, \mathbb{H})\) is
the vector space of all \(n \times n\) quaternionic matrices \(X\)
such that \(\bar{X}^T = - X\).</font>
</p>
</div>
<div style="margin-top: 0.5em; margin-left: 35.145870328812px">
<p>
<font style="font-size: 84.1%"><strong>题目 <class style="font-style: normal">2.4</class>.
</strong>Lie algebra of a complexx symplectic group</font>
</p>
</div>
<div style="margin-bottom: 0.5em; margin-left: 35.145870328812px">
<font style="font-size: 84.1%"><ol>
<li>
<p>
Show that the tangent space at the identity \(I\) of
\(\operatorname{Sp} (2 n, \mathbb{C}) \subset \operatorname{GL} (2
n,
\mathbb{C})\) is the vector space of all \(2 n \times 2 n\)
complex matrices \(X\) such that \(J X\) is symmetric.
</p>
</li>
<li>
<p>
Calculate the dimension of \(\operatorname{Sp} (2 n,
\mathbb{C})\).
</p>
</li>
</ol></font>
</div>
<p>
Notes taken and exported by TeXmacs.
</p>
  </body>
