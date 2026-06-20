---
layout: post
title: The Monge-Ampère Equation, Connor Mooney
date: 2026-06-20 00:02:00
description: this paper
tags: OT
categories: math
---

<body>
    <table class="title-block" style="margin-bottom: 2em">
      <tr>
        <td><table class="title-block" style="margin-top: 0.5em; margin-bottom: 0.5em">
          <tr>
            <td><font style="font-size: 168.2%"><strong>The Monge-Amp&egrave;re
            Equation</strong></font></td>
          </tr>
        </table><div class="compact-block" style="margin-top: 1em; margin-bottom: 1em">
          <table class="title-block">
            <tr>
              <td><p style="margin-top: 0.5em; margin-bottom: 0.5em">
                <div style="display: inline">
                  <span style="margin-left: 0pt"></span>
                </div>
                <table style="display: inline-table; vertical-align: middle">
                  <tbody><tr>
                    <td style="text-align: center; padding-left: 0em; padding-right: 0em; padding-bottom: 0em; padding-top: 0em; width: 100%"><center>
                      <p>
                        <class style="font-variant: small-caps">by Connor Mooney</class>
                      </p>
                    </center></td>
                  </tr></tbody>
                </table>
              </p><p style="margin-top: 0.5em; margin-bottom: 0.5em">
                <div style="display: inline">
                  <span style="margin-left: 0pt"></span>
                </div>
                <table style="display: inline-table; vertical-align: middle">
                  <tbody><tr>
                    <td style="text-align: center; padding-left: 0em; padding-right: 0em; padding-bottom: 0em; padding-top: 0em; width: 100%"><center>
                      <p>
                        2018
                      </p>
                    </center></td>
                  </tr></tbody>
                </table>
              </p></td>
            </tr>
          </table>
        </div></td>
      </tr>
    </table>
    <h2 id="auto-1">1<span style="margin-left: 1em"></span>Introduction<span style="margin-left: 1em"></span></h2>
    <p>
      Discuss the basic properties of Alexandrov solutions to the
      Monge-Amp&egrave;re equation.
    </p>
    <p>
      We then discuss the interior and boundary regularity for Alexandrov
      solutions to \(\det D^2 u = 1\).
    </p>
    <h2 id="auto-2">2<span style="margin-left: 1em"></span>Motivation<span style="margin-left: 1em"></span></h2>
    <p>
      The Monge-Amp&egrave;re equation
    </p>
    <center>
      \(\displaystyle \det D^2 u = f (x, u, \nabla u)\)
    </center>
    <p>
      for a convex function \(u\) on \(\mathbb{R}^n\) arises in several
      interesting applications.
    </p>
    <h3 id="auto-3">2.1<span style="margin-left: 1em"></span>Prescribed Gauss Curvature<span style="margin-left: 1em"></span></h3>
    <p>
      The Gauss curvature \(K (x)\) of the graph of a function \(u\) on
      \(\mathbb{R}^n\) at \((x, u (x))\) is given by
    </p>
    <center>
      \(\displaystyle \det D^2 u = K (x) (1 + | \nabla u |^2)^{\frac{n +
      2}{2}}\)
    </center>
    <p>
      It is a good exercise to derive this formula.
    </p>
    <p>
      高斯曲率 \(K\)
      的定义是形变算子（Shape
      Operator）的行列式，也可以表示为第二基本形式的行列式与第一基本形式的行列式之比：
    </p>
    <center>
      \(\displaystyle K = \frac{\det (h_{ij})}{\det (g_{ij})}\)
    </center>
    <p>
      参数化与切空间
    </p>
    <p>
      将函数 \(u : \mathbb{R}^n \to \mathbb{R}\)
      的图像看作 \(\mathbb{R}^{n + 1}\)
      中的一个 \(n\)
      维超曲面，其参数化映射可以写为：
    </p>
    <center>
      \(\displaystyle X (x) = (x_1, x_2, \ldots, x_n, u (x))\)
    </center>
    <p>
      对每个坐标 \(x_i\)
      求偏导，得到切空间的基向量：
    </p>
    <center>
      \(\displaystyle X_i = \frac{\partial X}{\partial x_i} = (0, \ldots, 1,
      \ldots, 0, u_i) = e_i +
u_i e_{n + 1}\)
    </center>
    <p>
      其中 \(u_i = \frac{\partial u}{\partial x_i}\) 是
      \(u\) 的偏导数，\(e_i\) 是
      \(\mathbb{R}^n\)
      的标准基向量。
    </p>
    <p>
      计算第一基本形式（度量张量）
    </p>
    <p>
      第一基本形式的系数
      \(g_{ij}\)
      是切向量的内积：
    </p>
    <center>
      \(\displaystyle g_{ij} = \langle X_i, X_j \rangle = \delta_{ij} + u_i
      u_j\)
    </center>
    <p>
      为了求出矩阵 \((g_{ij})\)
      的行列式 \(\det
      (g_{ij})\)，我们可以利用矩阵行列式引理（Matrix
      Determinant Lemma），即 \(\det (I + vv^T) = 1 +
      |v|^2\)。在这里，该矩阵可以表示为单位矩阵加上梯度向量的特征线性组合，因此：
    </p>
    <center>
      \(\displaystyle \det (g_{ij}) = 1 + | \nabla u|^2\)
    </center>
    <p>
      计算单位法向量
    </p>
    <p>
      超曲面的单位法向量
      \(\nu\)
      需要同时垂直于所有的切向量
      \(X_i\)，且长度为
      1。容易验证，未归一化的法向量可以取为
      \((- \nabla u, 1)\)，因为：
    </p>
    <center>
      \(\displaystyle \langle X_i, (- \nabla u, 1) \rangle = - u_i + u_i = 0\)
    </center>
    <p>
      对其进行归一化后，得到单位法向量：
    </p>
    <center>
      \(\displaystyle \nu = \frac{(- \nabla u, 1)}{\sqrt{1 + | \nabla u|^2}}\)
    </center>
    <p>
      计算第二基本形式
    </p>
    <p>
      对切向量再次求导，得到二阶偏导数位置向量：
    </p>
    <center>
      \(\displaystyle X_{i j} = \frac{\partial^2 X}{\partial x_i \partial x_j}
      = (0, 0, \ldots, 0,
u_{i j})\)
    </center>
    <p>
      其中 \(u_{ij} = \frac{\partial^2 u}{\partial x_i \partial
      x_j}\) 是 Hessian 矩阵 \(D^2 u\)
      的元素。
    </p>
    <p>
      第二基本形式的系数
      \(h_{ij}\) 定义为 \(X_{ij}\)
      在法方向 \(\nu\)
      上的投影：
    </p>
    <center>
      \(\displaystyle h_{ij} = \langle X_{ij}, \nu \rangle =
      \frac{u_{ij}}{\sqrt{1 + | \nabla u|^2}}\)
    </center>
    <p>
      写成矩阵形式，第二基本形式矩阵为：
    </p>
    <center>
      \(\displaystyle H = \frac{D^2 u}{\sqrt{1 + | \nabla u|^2}}\)
    </center>
    <p>
      因为 \(H\) 是一个 \(n \times n\)
      的矩阵，当提取常数标量因子时，行列式需要提取该因子的
      \(n\) 次方：
    </p>
    <center>
      \(\displaystyle \det (h_{ij}) = \det \left( \frac{D^2 u}{\sqrt{1 + |
      \nabla u|^2}} \right) =
\frac{\det D^2 u}{(1 + | \nabla u|^2)^{n / 2}}\)
    </center>
    <p>
      结合高斯曲率定义
    </p>
    <p>
      根据微分几何，超曲面的高斯曲率
      \(K (x)\) 等于：
    </p>
    <center>
      \(\displaystyle K = \frac{\det (h_{ij})}{\det (g_{ij})}\)
    </center>
    <p>
      将步骤 2 和步骤 4
      中得到的行列式代入上式中：
    </p>
    <center>
      \(\displaystyle K = \frac{\det D^2 u}{(1 + | \nabla u |^2)^{(n + 2) /
      2}}\)
    </center>
    <h3 id="auto-4">2.2<span style="margin-left: 1em"></span>Optimal Transport<span style="margin-left: 1em"></span></h3>
    <p>
      Given probability densities \(f, g\) supported on domains \(\Omega_f,
      \Omega_g\) in \(\mathbb{R}^n\), the optimal transport problem asks to
      minimize the transport cost
    </p>
    <center>
      \(\displaystyle J (T) = \int_{\Omega_f} | T (x) - x |^2 f (x) d x\)
    </center>
    <p>
      over measure-preserving maps \(T : \Omega_f \rightarrow \Omega_g\) (that
      is \(f (x) d x = g (T (x)) \det D T (x) d x\)). An important theorem of
      Brenier says that the optimal map exists, and is given by the gradient
      of a convex function \(u\) on \(\Omega_f\). The measure-preserving
      condition implies
    </p>
    <center>
      \(\displaystyle \det D^2 u = \frac{f (x)}{g (\nabla u (x))},\)
    </center>
    <p>
      in a certain weak sense.
    </p>
    <h3 id="auto-5">2.3<span style="margin-left: 1em"></span>Fluid Dynamics<span style="margin-left: 1em"></span></h3>
    <p>
      Large-scale fluid flows in \(\mathbb{R}^2\) are modeled by a system of
      evolution equations for a probability density \(\rho (x, t)\) and a
      function \(u (x, t)\) that is convex in \(x\) for all \(t\). The system
      is
    </p>
    <center>
      \(\displaystyle \left\{ \begin{array}{l}
  \partial_t \rho + (x - \nabla
      u)^{\bot} \cdot \nabla \rho = 0\\
  \det D^2 u = \rho .
\end{array}
      \right.\)
    </center>
    <p>
      Here \(w^{\bot}\) denotes the counter-clockwise rotation of \(w\) by
      \(\frac{\pi}{2}\). This can be viewed as a fully nonlinear version of
      the incompressible Euler equations in 2\(D\), where the
      Monege-Amp&egrave;re operator repalces the Laplace operator.
    </p>
    <p>
      
    </p>
    <h2 id="auto-6">3<span style="margin-left: 1em"></span>Weak Solutions<span style="margin-left: 1em"></span></h2>
    <p>
      We introduce a useful notion of weak solution based on the idea of
      polyhedral approximations. We then solve the Dirichlet problem on
      bounded domains. Detailed expositions of these topics can be found in
      work of Cheng-Yau, and in the books of Gutierrez and Figalli.
    </p>
    <h3 id="auto-7">3.1<span style="margin-left: 1em"></span>Alexandrov Solutions<span style="margin-left: 1em"></span></h3>
    <p>
      If \(v\) is a \(C^2\) convex function on \(\mathbb{R}^n\), then the area
      formula gives
    </p>
    <center>
      \(\displaystyle \int_{\Omega} \det D^2 v d x = | \nabla v (\Omega) | .\)
    </center>
    <p>
      For an arbitrary convex function \(v\) on a domain \(\Omega \subset
      \mathbb{R}^n\) and \(E \subset \Omega\) we define
    </p>
    <center>
      \(\displaystyle M v (E) = | \partial v (E) |,\)
    </center>
    <p>
      where \(\partial v (E)\) is the set of slopes of supporting hyperplanes
      to the graph of \(v\) (the sub-gradients of \(v\)) over points in \(E\).
      We have ([F], Theorem 2.3)
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">1</class>. </strong><i>\(M v\) is
      a Borel measure on \(\Omega\).</i>
    </p>
    <p>
      We call \(M v\) the Monge-Amp&egrave;re measure of \(v\).
    </p>
    <p>
      It is easy to check that if \(v \in C^2\), then \(M v = \det D^2 v d
      x\). A more interesting example is the polyhedral graph
    </p>
    <center>
      \(\displaystyle v = \max_{1 \leq i \leq 3} \{ p_i \cdot x \}\)
    </center>
    <p>
      over \(\mathbb{R}^2\). The set \(\partial v (0)\) is the (closed)
      triangle with vertices \(\{ p_i \}\). The sub-gradients of the
      &quot;edges&quot; of the graph are the segments joining \(p_i\), and the
      sub-gradients of the &quot;faces&quot; are \(p_i\). Thus, \(M v\) is a
      Dirac mass at 0 with weight given by the area of the triangle.
    </p>
    <p>
      
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">1</class>. </strong><i>Let
      \(\mu\) be a Borel measure on a domain \(\Omega \subset \mathbb{R}^n\).
      We say that a convex function \(u\) on \(\Omega\) is an Alexandrov
      solution of \(\det D^2 u = \mu\) if \(M u = \mu\).</i>
    </p>
    <p>
      The key fact that is used to prove Proposition 1, and is essential for
      many other parts of the theory, is ([F], Lemma A.30):
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">2</class>. </strong><i>Let \(v\)
      be a convex function on a domain \(\Omega \subset \mathbb{R}^n\).
      Then</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle | \{ p \in \mathbb{R}^n : p \in \partial v (x) \cap
        \partial v (y)
\operatorname{for}\operatorname{some}x \neq y \in
        \Omega \} | = 0.\)
      </center></i>
    </p>
    <h3 id="auto-8">3.2<span style="margin-left: 1em"></span>Maximum Principle and Compactness<span
    style="margin-left: 1em"></span></h3>
    <p>
      Alexandrov solutions are useful because they satisfy a maximum principle
      and have good compactness properties.
    </p>
    <p>
      We first observe that if \(u\) and \(v\) are convex on a bounded domain
      \(\Omega\), with \(u = v\) on \(\partial \Omega\) and \(u \leq v\) in
      \(\Omega\), then
    </p>
    <center>
      \(\displaystyle \partial v (\Omega) \subset \partial u (\Omega) .\)
    </center>
    <p>
      From this observation one concludes the comparison principle ([F],
      Theorem 2.10):
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">3</class>. </strong><i>Assume
      \(u\) and \(v\) are convex on a bounded domain \(\Omega\), with \(u =
      v\) on \(\partial \Omega\). If \(M u \geq M v\) in \(\Omega\), then \(u
      \leq v\) in \(\Omega\).</i>
    </p>
    <p>
      Alexandrov maximum principle ([F], Theorem 2.8):
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">4</class>. </strong><i>If \(u\)
      is convex on a bounded convex domain \(\Omega\) and \(u |_{\partial
      \Omega} = 0\), then</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle | u (x) | \leq C (n, \operatorname{diam} (\Omega), M u
        (\Omega))
\operatorname{dist}. (x, \partial \Omega)^{1 / n} .\)
      </center></i>
    </p>
    <p>
      This says that functions with bounded Monge-Amp&egrave;re mass have a
      \(C^{1 / n}\) modulus of continuity near the boundary of a sub level
      set, that depends only on rough geometric properties of this set.
    </p>
    <p>
      证明思路略去。
    </p>
    <p>
      The other important property of Alexandrov solutions is closedness under
      uniform convergence. That is:
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">5</class>. </strong><i>If \(u_k\)
      converge uniformly to \(u\) in \(\Omega \subset \mathbb{R}^n\), then \(M
      u_k\) converges weakly to \(M u\) in \(\Omega\).</i>
    </p>
    <p>
      For a detailed proof see [F], Proposition 2.6.
    </p>
    <p>
      As a consequence of Proposition 4 and 5 we have the compactness of
      solutions with fixed linear boundary data:
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">6</class>. </strong><i>For a
      bounded convex domain \(\Omega\), the collection of functions</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \mathcal{A}= \{
        v\operatorname{convex}\operatorname{on} \Omega, v |_{\partial
\Omega}
        = 0, M v (\Omega) \leq C_0 \}\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>is compact. That is, any sequence in \(\mathcal{A}\) has a uniformly
      convergent subsequence whose Monge-Amp&egrave;re measures converge
      weakly to that of the limit.</i>
    </p>
    <h3 id="auto-9">3.3<span style="margin-left: 1em"></span>Dirichlet Problem<span style="margin-left: 1em"></span></h3>
    <p>
      We conclude this section by discussing the Dirichlet problem.
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">1</class>. </strong><i>Let
      \(\Omega \subset \mathbb{R}^n\) be a bounded strictly convex domain,
      \(\mu\) a bounded Borel measure on \(\Omega\), and \(\varphi \in C
      (\partial \Omega)\). Then there exists a unique Alexandrov solution in
      \(C (\bar{\Omega})\) to the Dirichlet problem</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \left\{ \begin{array}{ll}
  \det D^2 u = \mu &
        \operatorname{in} \Omega\\
  u |_{\partial \Omega} = \varphi . &
        
\end{array} \right.\)
      </center></i>
    </p>
    <p>
      <div style="display: inline">
        <a id="auto-10"></a>
      </div>
      <h5>Sketch of Proof<span style="margin-left: 1em"></span></h5>
      <div style="display: inline">
        Refer the reader to [CY] for details.
      </div>
    </p>
    <p style="margin-top: 1em">
      <strong>注意 <class style="font-style: normal">1</class>. </strong>When
      \(\varphi\) is linear, we don't require strict convexity of \(\partial
      \Omega\). The strict convexity is necessary for general \(\varphi\)
      since no convex function can continuously attain e.g. the boundary data
      \(- | x |^2\) when \(\partial \Omega\) has flat pieces.
    </p>
    <p>
      The strict convexity of \(\partial \Omega\) is used in the last step. It
      guarantees that for any subset \(\{ y_i \}_{i = 1}^M\) of \(\partial
      \Omega\), each \(y_k\) is a vertex of the convex hull of \(\{ y_i \}_{i
      = 1}^M\).
    </p>
    <p style="margin-bottom: 1em">
      Closely related is the fact that when \(\partial \Omega\) is strictly
      convex, the convex envelope of the graph of \(\varphi\) (that is, the
      supremum of linear functions beneath the graph) continuously achieves
      the boundary data, and has 0 Monge-Amp&egrave;re measure. 
    </p>
    <p>
      
    </p>
  </body>
