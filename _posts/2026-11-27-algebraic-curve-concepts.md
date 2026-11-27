---
layout: post
title: Algebraic Curves Basic Concepts
date: 2026-11-27 20:50:00 +0800
description: Algebraic Curves, Griffith
tags: Riemann surfaces
categories: math
tikzjax: true
---

<!-- Preserve exported HTML and LaTeX: Markdown formatting can rewrite math operators. -->

<!-- prettier-ignore-start -->

<body>
    <table class="title-block" style="margin-bottom: 2em">
      <tr>
        <td><table class="title-block" style="margin-top: 0.5em; margin-bottom: 0.5em">
          <tr>
            <td><font style="font-size: 168.2%"><strong>Algebraic Curves</strong></font></td>
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
                        <class style="font-variant: small-caps">by P.
                        格列菲斯</class>
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
    <h1 id="auto-1">1<span style="margin-left: 1em"></span>基本概念<span style="margin-left: 1em"></span></h1>
    <h2 id="auto-2">1<span style="margin-left: 1em"></span>复射影平面\(P^2
    \mathbb{C}\)上的代数曲线<span style="margin-left: 1em"></span></h2>
    <p>
      设 \(f (x, y)\)
      是二元实系数多项式。在
      \(\mathbb{R}^2\) 上由方程
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle f (x, y) = 0\)</td>
        <td align="right">(1.1)</td>
      </tr>
    </table>
    <p>
      给出的图形称为实代数曲线；次数称为这代数曲线的次数。一次和二次的实代数曲线包括直线、椭圆、双曲线、抛物线等。但是，限制在实数范围内讨论更一般的代数曲线，会有若干不方便之处，难以获得整齐的结果。主要是因为实数域不是代数封闭域。例如，我们欲讨论这样的问题：一条任意直线\(L\)与由(1.1)给出的代数曲线
      \(C\)
      有多少交点。不失一般性，可设\(L\)通过坐标原点，它的参数方程可以写成
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \left\{ \begin{array}{l}
  x =
        \alpha t,\\
  y = \beta t.
\end{array} \right.\)</td>
        <td align="right">(1.2)</td>
      </tr>
    </table>
    <p>
      又设
    </p>
    <center>
      \(\displaystyle f (x, y) = f_n (x, y) + f_{n - 1} (x, y) + \cdots +
      f_0,\)
    </center>
    <p>
      其中 \(f_k (x, y)\)
      是\(k\)次齐次多项式。把(2)代入(1)得
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle f_n (\alpha, \beta) t^n + f_{n - 1}
        (\alpha, \beta) t^{n - 1} + \cdots + f_0 =
0\)</td>
        <td align="right">(1.3)</td>
      </tr>
    </table>
    <p>
      在实数范围内要判断这样的方程有多少根，不是一件轻而易举的事，而且结论也很不整齐。但是，如果考虑复系数的二元多项式\(f
      (x,
      y)\)而把(1.1)视为\(\mathbb{C}^2\)上的代数曲线，再来讨论复直线与复代数曲线的交点，也得到(1.3)。由熟知的代数基本定理，只要
      \(f_n (\alpha, \beta) \neq
      0\)，方程(1.3)恰有\(n\)个根（重根重复计数），也就是有\(n\)个交点。上面的讨论中还有一种例外的情形：如果
    </p>
    <center>
      \(\displaystyle f_n (\alpha, \beta) = \cdots = f_{m + 1} (\alpha {,}
      \beta) = 0,\)
    </center>
    <center>
      \(\displaystyle f_m (\alpha, \beta) \neq 0,\)
    </center>
    <p>
      那么\(L\)和\(C\)在\(\mathbb{C}^2\)上将只有\(m\)个交点。这时可以认为有\(n
      -
      m\)个交点在无穷远处。
    </p>
    <p>
      为了方便讨论，需要给\(\mathbb{C}^2\)添加无穷远直线以构成复射影平面\(P^2
      \mathbb{C}\)。本书讨论的主要对象，将是复射影平面上的代数曲线。
    </p>
    <p>
      添加无穷远直线，方便的办法是通过<b>齐次坐标</b>。对于点\((x,
      y) \in \mathbb{C}^2\)，其齐次坐标
      \((\zeta, \xi, \eta)\)是满足下式
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle x = \frac{\xi}{\zeta}, \quad y =
        \frac{\eta}{\zeta}\)</td>
        <td align="right">(1.4)</td>
      </tr>
    </table>
    <p>
      的复数组。显然 \((\lambda
      \zeta, \lambda \xi, \lambda \eta)\) \((\lambda \in \mathbb{C}, \lambda
      \neq 0)\)也是 \((x,
      y)\)的齐次坐标。
    </p>
    <p>
      要使(1.4)有意义，必须
      \(\zeta \neq 0\)。但如果 \(\xi, \eta\)
      不全为0，当 \(\zeta \rightarrow 0\)
      时，点\((x, y)\)沿\(\xi :
      \eta\)方向趋向无穷远，因而我们认为\((0,
      \xi, \eta)\)表示 \(\xi : \eta\)
      方向的无穷远点。这与，通过齐次坐标，我们给\(\mathbb{C}^2\)中每个方向添加了一个无穷远
      点。所有的无穷远点组成一条无穷远直线
      \(L_{\infty} : \zeta =
      0\)。添加了无穷远直线的
      \(\mathbb{C}^2\)
      称为是复射影平面 \(P^2
      \mathbb{C}\)。下面更严格地叙述这一过程。
    </p>
    <p>

    </p>
    <p>
      在集合 \(\mathbb{C}^2 \backslash \{ (0, 0, 0)
      \}\)中建立一个等价关系
      &quot;\(\sim\)&quot;:
    </p>
    <p>
      \((\zeta, \xi, \eta) \sim (\zeta', \xi',
      \eta')\)当且仅当存在 \(\lambda \in
      \mathbb{C}, \lambda \neq 0\)，使得
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \zeta' = \lambda \zeta, \xi' =
        \lambda \xi, \eta' = \lambda \eta .\)</td>
        <td align="right">(1.5)</td>
      </tr>
    </table>
    <p>
      这是一个等价关系。
    </p>
    <p>
      \((\zeta, \xi, \eta) \in \mathbb{C}^2 \backslash \{ (0, 0, 0)
      \}\)所在的等价类记为\([\zeta,
      \xi, \eta]\). 显然有
    </p>
    <center>
      \(\displaystyle [\zeta, \xi, \eta] = [\lambda \zeta, \lambda \xi,
      \lambda \eta] \quad \forall
\lambda \in \mathbb{C}\backslash \{ 0 \} .\)
    </center>
    <p>
      按这等价关系作成的商空间（等价类的空间）
    </p>
    <center>
      \(\displaystyle (\mathbb{C}^2 \backslash \{ (0, 0, 0) \}) / \sim\)
    </center>
    <p>
      称为复射影平面，记为
      \(P^2
      \mathbb{C}\)。作为商空间，\(P^2
      \mathbb{C}\)带有商拓扑结构。
    </p>
    <p>

    </p>
    <p>
      考察(1.1)给出的曲线\(C\)在齐次坐标下的表示。把

    </p>
    <center>
      \(\displaystyle x = \frac{\xi}{\zeta}, y = \frac{\eta}{\zeta}\)
    </center>
    <p>
      代入(1.1)并以 \(\zeta^n\)
      乘以两边得
    </p>
    <center>
      \(\displaystyle F (\zeta, \xi, \eta) = f_n (\xi, \eta) + f_{n - 1} (\xi,
      \eta) \zeta + \cdots
+ f_0 \zeta^n = 0\)
    </center>
    <p>
      这方程的左端是一个关于
      \(\zeta, \xi,
      \eta\)的齐次多项式。
    </p>
    <p>
      一般地，如果 \(F (\zeta, \xi,
      \eta)\)是关于 \(\zeta, \xi, \eta\)
      的齐次多项式，那么

    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle F (\zeta, \xi, \eta) = 0\)</td>
        <td align="right">(1.6)</td>
      </tr>
    </table>
    <p>
      表示 \(P^2 \mathbb{C}\)
      上的一条代数曲线，\(F\)的次数称为这曲线的次数。(1.6)称为这曲线的齐次方程。限制在
      \(\mathbb{C}^2 = P^2 \mathbb{C}\backslash
      L_{\infty}\)上，这曲线满足仿射方程
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle f (x, y) = 0\)</td>
        <td align="right">(1.7)</td>
      </tr>
    </table>
    <p>
      其中
    </p>
    <center>
      \(\displaystyle f (x, y) = F (1, x, y) .\)
    </center>
    <p>
      这样，一条曲线的齐次方程决定这曲线在\(\mathbb{C}^2
      = P^2 \mathbb{C}\backslash
      L_{\infty}\)上的仿射方程。反过来，曲线的次数\(n\)和它在\(\mathbb{C}^2
      = P^2 \mathbb{C}\backslash
      L_{\infty}\)上的仿射方程\(f (x, y) =
      0\)也唯一的决定它的齐次方程
    </p>
    <center>
      \(\displaystyle F (\zeta, \xi, \eta) = 0.\)
    </center>
    <p>
      这里
    </p>
    <center>
      \(\displaystyle F (\zeta, \xi, \eta) = \zeta^n f \left(
      \frac{\xi}{\zeta}, \frac{\eta}{\zeta}
\right) .\)
    </center>
    <p>
      如果代数曲线\(C\)由方程
    </p>
    <center>
      \(\displaystyle F (\zeta, \xi, \eta) = 0\)
    </center>
    <p>
      给出，而\(F\)分解为不可约齐次多项式的乘积
    </p>
    <center>
      \(\displaystyle F = F_1^{m_1} \cdot F_2^{m_2} \ldots F_l^{m_l},\)
    </center>
    <p>
      则记
    </p>
    <center>
      \(\displaystyle C = m_1 C_1 + \cdots + m_l C_l,\)
    </center>
    <p>
      这里
    </p>
    <center>
      \(\displaystyle C_j = \{ (\zeta, \xi, \eta) \in P^2 \mathbb{C}|F_j
      (\zeta, \xi, \eta) = 0 \},
j = 1, 2, \ldots, l\)
    </center>
    <p>
      称为是\(C\)的不可约分支。特别地，如果\(F\)本身是不可约的，则称\(C\)为不可约曲线。
    </p>
    <h2 id="auto-3">2<span style="margin-left: 1em"></span>Riemann面<span style="margin-left: 1em"></span></h2>
    <p>
      紧Riemann面与代数曲线的研究有密切的关系。具体来说就是，任何不可约的平面代数曲线，都有全纯的参数表示，这参数定义的定义域是一紧Riemann面。更准确的表述，是下面的正则化定理。
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">2.1</class>.
      </strong><i>对任一不可约代数曲线\(C
      \subset P^2
      \mathbb{C}\)，存在一个紧Riemann面\(\tilde{C}\)和一个全纯映射</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \sigma : \tilde{C} \rightarrow P^2 \mathbb{C},\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>使得 \(\sigma (\tilde{C}) = C\)，并且
      \(\sigma\) 在
      \(C\)的光滑点之上是一对一的。</i>
    </p>
    <p>
      这样的紧Riemann面 \(\tilde{C}\)
      连同全纯映射 \(\sigma\)
      称为是 \(C\)
      的正则化(Normalization)。
    </p>
    <p>
      另一方面，任何紧Riemann面也都可以表示为代数曲线。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">2.2</class>.
      </strong><i>任意紧 Riemann 面 \(\tilde{C}\)
      都可以作为某平面代数曲线
      \(C\)
      的正则化，并可要求这代数曲线
      \(C\)
      至多只具有通常二重点。这就是说，存在全纯映射
      \(\sigma : \tilde{C} \rightarrow P^2 \mathbb{C}\),使得
      \(\sigma (\tilde{C})\)
      是一条至多只有通常二重点的代数曲线。</i>
    </p>
    <p>
      证明要用到层的上同调理论。
    </p>
    <p>
      由此看来，紧Riemann面的研究和代数曲线的研究，实际上是一回事，都是贯穿本书的主要内容。
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">2.3</class>.
      </strong><i>一个Riemann面是一个连通的Hausdorff拓扑空间
      \(C\), 连同 \(C\)
      的一族开覆盖 \(\{ U_{\alpha}
      \}\)和一族映射</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle z_{\alpha} : U_{\alpha} \rightarrow \mathbb{C},\)
      </center></i>
    </p>
    <p>
      <i>满足以下条件：</i>
    </p>
    <i><ol>
      <li>
        <p>
          \(z_{\alpha} : U_{\alpha} \rightarrow \mathbb{C}\) 是到
          \(\mathbb{C}\) 中开集 \(z_{\alpha} (U_{\alpha})\)
          的同胚映射；
        </p>
      </li>
      <li>
        <p>
          只要 \(U_{\alpha} \cap U_{\beta} \neq
          \varnothing\)，函数
        </p>
        <center>
          \(\displaystyle z_{\beta} \circ z_{\alpha}^{- 1} : z_{\alpha}
          (U_{\alpha} \cap U_{\beta})
\rightarrow z_{\beta} (U_{\alpha} \cap
          U_{\beta})\)
        </center>
        <p>
          就是双全纯的。
        </p>
      </li>
    </ol></i>
    <p style="margin-bottom: 1em">
      <i>这时我们把 \((U_{\alpha},
      z_{\alpha})\)叫作局部全纯坐标，而把
      \(\{ (U_{\alpha}, z_{\alpha})
      \}\)称为全纯坐标覆盖。</i>
    </p>
    <figure style="margin: 1em 0; overflow-x: auto; text-align: center" aria-label="局部全纯坐标之间的转换交换图">
      <script type="text/tikz" data-tex-packages='{ "amsmath": "", "amssymb": "", "amsfonts": "", "tikz-cd": "" }'>
        \begin{tikzcd}
          & U_{\alpha} \cap U_{\beta}
            \arrow[rd, "z_{\beta}"] \arrow[ld, "z_{\alpha}"'] & \\
          z_{\alpha}(U_{\alpha} \cap U_{\beta})
            \arrow[rr, "z_{\beta} \circ z_{\alpha}^{- 1}"]
          & & z_{\beta}(U_{\alpha} \cap U_{\beta})
        \end{tikzcd}
      </script>
      <figcaption>重叠区域上的坐标变换为双全纯映射。</figcaption>
    </figure>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>扩充复数 \(\Sigma
      =\mathbb{C} \cup \{ \infty \}\).
    </p>
    <p>
      通过球极投影可以把单位球面
      \(S\) 与 \(\Sigma\)
      等同起来，因为 \(S\)
      也自然地称为 Riemann
      面。扩充的复平面
      \(\Sigma\)
      或通过球极投影得到的复数球面
      \(S\)，按上面的方式作成的紧
      Riemann 面称为是 Riemann
      球面。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>复环面 \(C =\mathbb{C}/
      \Lambda\).
    </p>
    <p>
      设复数 \(w_1,
      w_2\)实线性无关，它们在
      \(\mathbb{C}\)
      上定义了一个格：
    </p>
    <center>
      \(\displaystyle \Lambda = \{ m_1 w_1 + m_2 w_2 |m_1, m_2 \in \mathbb{Z}
      \} \subset \mathbb{C}.\)
    </center>
    <p>
      \(A\) 是由 \(w_1, w_2\) 生成的
      \(\mathbb{C}\)
      的离散子群，它同构于
      \(\mathbb{Z} \oplus \mathbb{Z}\)。考虑 \(\mathbb{C}\)
      上的等价关系\(\sim\)：
    </p>
    <center>
      \(\displaystyle z \sim z' \Leftrightarrow z - z' \in \Lambda .\)
    </center>
    <p>
      对这等价关系的商空间
      \(C =\mathbb{C}/
      \Lambda\)称为复环面。
    </p>
    <p>
      从\(\mathbb{C}\)的开集\(U\)到\(\mathbb{C}\)的开集\(V\)的全纯函数
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle w = f (z) \quad (w = u + i v, z = x
        + i y),\)</td>
        <td align="right">(2.1)</td>
      </tr>
    </table>
    <p>
      同时也给出从
      \(\mathbb{R}^2\)的开集 \(U\)到
      \(\mathbb{R}^2\)的开集
      \(V\)的光滑映射
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \left\{ \begin{array}{l}
  u = u (x,
        y),\\
  v = v (x, y) .
\end{array} \right.\)</td>
        <td align="right">(2.2)</td>
      </tr>
    </table>
    <p>
      由此看出紧Riemann面同时也是一个紧的光滑的实二维流形&mdash;紧光滑曲面。由全纯函数
      (2.1)
      引出的光滑映射(2.2)应满足Cauchy-Riemann方程
    </p>
    <center>
      \(\displaystyle \left\{ \begin{array}{l}
  \frac{\partial u}{\partial x}
      = \frac{\partial v}{\partial y},\\
  \frac{\partial u}{\partial y} = -
      \frac{\partial v}{\partial x} .
\end{array} \right.\)
    </center>
    <p>
      因而
    </p>
    <center>
      \(\displaystyle \det \left| \begin{array}{cc}
  \frac{\partial
      u}{\partial x} & \frac{\partial u}{\partial y}\\
  \frac{\partial
      v}{\partial x} & \frac{\partial v}{\partial y}
\end{array} \right| =
      \left( \frac{\partial u}{\partial x} \right)^2 + \left(
\frac{\partial
      u}{\partial y} \right)^2 > 0.\)
    </center>
    <p>
      这说明紧Riemann面表示的光滑的二维实流形是可定向的。从拓扑上看，任何紧Riemann面同时也是一个可定向的二维紧光滑曲面。
    </p>
    <p>
      任何可定向的二维紧流形都同胚于一个具有若干环柄的球面；这环柄的个数\(g\)，是一个基本的拓扑不变量，称为亏格。
    </p>
    <p>
      一般地，具有\(g\)个环柄的球面，可以沿\(2
      g\)条割缝\(a_1, b_1, \ldots, a_g, b_g\)
      切开，摊成平面上的\(4
      g\)边形:
    </p>
    <center>
      \(\displaystyle a_1 b_1 a_{1 }^{- 1} b_1^{- 1} \ldots a_g b_g a_g^{- 1}
      b_g^{- 1} .\)
    </center>
    <p>
      这种表示法称为规范表示。
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">2.4</class>.
      </strong><i>对于亏格为\(g\)的二维可定向紧曲面，其Euler示性数定义为</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \boldsymbol{\chi}= 2 - 2 g.\)
      </center></i>
    </p>
    <h2 id="auto-4">3<span style="margin-left: 1em"></span>全纯与半纯函数<span
    style="margin-left: 1em"></span></h2>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">3.1</class>.
      </strong><i>设\(C\)是一个Riemann面，\(\{
      (U_{\alpha}, z_{\alpha})
      \}\)是它的全纯坐标覆盖，\(C\)上的一个半纯函数（全纯函数）\(f\)，由一族映射\(f_i
      : U_i \rightarrow \mathbb{C} (\forall
      i)\)给出，满足</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            只要 \(U_i \cap U_j \neq \varnothing\),在 \(U_i
            \cap U_j\) 上就有
          </p>
          <center>
            \(\displaystyle f_i = f_j ;\)
          </center>
        </li>
        <li>
          <p>
            \(\forall i, f_i \circ z_i^{- 1}\)
            都是半纯（全纯）函数。
          </p>
        </li>
      </ol></i>
    </div>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">3.2</class>.
      </strong>Riemann面\(C\)上的所有半纯函数，按照通常方法定义的加法和乘法，形成一个域，称为\(C\)上的半纯函数域，记为
      \(K
      (C)\)。\(C\)上的所有全纯函数，按照通常方式定义的加法、乘法和数乘，成为一个\(C\)上的代数，记为
      \(\mathcal{O} (C)\)。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">3.3</class>.
      </strong><i>紧Riemann面\(C\)上的全纯函数，只能是常值函数。</i>
    </p>
    <p>

    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">3.4</class>.
      </strong><i>设\(C\)是一个紧Riemann面，\(f
      \in K (C), p \in
      C\)。取\(p\)点邻近的局部坐标\(z\)，使得
      \(z (p) = 0\)，则在\(p\)点附近</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle f = z^{\nu} h (z),\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>这里 \(h (z)\)
      是一全纯函数，\(h (0) \neq 0,
      \nu \in
      \mathbb{Z}\).上述\(\nu\)与局部坐标无关，由\(f\)所决定，称为\(f\)在\(p\)点的重数，记为
      \(\nu_p (f)\)。\(\nu_p (f) >
      0\)时，\(p\)称为\(f\)的零点；\(\nu_p
      (f) <
      0\)时，\(p\)称为\(f\)的极点。</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">3.5</class>.
      </strong><i>Riemann球面\(S\)上的半纯函数域
      \(K (S)\) 同构于 \(C\)
      上的有理分式域 \(C
      (z)\)。</i>
    </p>
    <p>

    </p>
    <p>
      对复环面，我们有以下结果。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">3.6</class>. </strong><i>\(K (C /
      \Lambda)\)同构于以 \((w_1,
      w_2)\)为周期的双周期半纯函数所成之域。</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">3.7</class>.
      </strong>根据椭圆函数论中的熟知结果，任何以
      \((w_1, w_2)\)
      为周期的双周期半纯函数，都可表示为具有同样双周期的Weierstrass椭圆函数
      \(\mathcal{P} (z)\)和 \(\mathcal{P}'
      (z)\)的有理分式。因而，\(K
      (C / \Lambda)\)由\(\mathcal{P} (z)\)，\(\mathcal{P}'
      (z)\)生成。
    </p>
    <p>
      之后在模形式部分可以展开。
    </p>
    <h2 id="auto-5">4<span style="margin-left: 1em"></span>全纯微分与半纯微分<span
    style="margin-left: 1em"></span></h2>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">4.1</class>.
      </strong><i>设\(C\)是一个Riemann面，\(C\)上的一个全纯微分（半纯微分）\(\omega\)
      由满足下面条件的一族
      \(\{ (U_i, z_i, \omega_i) \}\)给出</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><ol>
        <li>
          <p>
            \(\{ (U_i, z_i)
            \}\)是\(C\)的全纯坐标覆盖,
          </p>
          <center>
            \(\displaystyle \omega_i = f_i (z_i) \mathrm{d} z_i\)
          </center>
          <p>
            这里 \(f_i \in \mathcal{O} (U_i) (K (U_i))\);
          </p>
        </li>
        <li>
          <p>
            如果 \(z_i = \varphi_{i j} (z_j)\)是在
            \(U_i \cap U_j (\neq
            \varnothing)\)上的坐标变换，那么
          </p>
          <center>
            \(\displaystyle f_i (\varphi_{i j} (z_j)) \frac{\mathrm{d}
            \varphi_{i j} (z_j)}{\mathrm{d}
z_j} = f_j (z_j),\)
          </center>
          <p>
            即微分的局部表示按链式法则变换
          </p>
          <center>
            \(\displaystyle f_i (\varphi_{i j} (z_j)) \mathrm{d} \varphi_{i j}
            (z_j) = f_j (z_j)
\mathrm{d} z_j .\)
          </center>
        </li>
      </ol></i>
    </div>
    <p>

    </p>
    <p>
      \(C\)上全纯微分（半纯微分）的集合记为
      \(\Omega^1 (C) (K^1 (C))\)。
    </p>
    <p>
      以下结果是显然的。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>命题 <class style="font-style: normal">4.2</class>. </strong><i>设
      \(\omega_0,
      \omega_1\)是\(C\)上半纯微分，\(\omega_0
      \mathrel{\not\equiv} 0\)，则 \(\omega_1 /
      \omega_0\)给出\(C\)上一个半纯函数。</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>设\(r
      (z)\)是\(C\)上有理分式，则
      \(r (z) \mathrm{d}
      z\)给出Riemann球面\(S\)上的半纯微分。
    </p>
    <p>
      习题 试证所有的 \(\omega \in
      K^1
      (S)\)都具有上述形式。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例. </strong>设 \(\Lambda = \{ m_1 \omega_1 + m_2
      \omega_2 |m_1, m_2 \in \mathbb{Z}
      \}\)是\(\mathbb{C}\)上的一个格，\(f\)是以
      \((\omega_1,
      \omega_2)\)为周期的双周期半纯函数，则\(f
      (w) \operatorname{dw}\)是定义于\(C
      =\mathbb{C}\backslash
      \Lambda\)上的半纯微分。
    </p>
    <p style="margin-top: 1em">
      <strong>例.
      </strong>设\(C\)是一个Riemann面，
    </p>
    <center>
      \(\displaystyle f = \{ (U_i, z_i, f_i (z_i)) \} \in K (C) .\)
    </center>
    <p>
      则由下式定义了一个半纯微分\(\mathrm{d}
      f \in K^1 (C) :\)
    </p>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle \mathrm{d} f = \left\{ \left( U_i, z_i, \mathrm{d} f_i
        (z_i) =
\frac{\mathrm{d} f_i (z_i)}{\mathrm{d} z_i} \mathrm{d} z_i
        \right) \right\} .\)
      </center>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定义 <class style="font-style: normal">4.3</class>.
      </strong><i>由上式定义的\(\mathrm{d}
      f\)称为半纯函数\(f\)的微分。</i>
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">4.4</class>.
      </strong><i>设\(C\)是一个Riemann面，</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \omega = \{ (U_i, z_i, f_i (z_i) \operatorname{dz}_i)
        \} \in K^1 (C), \quad p
\in U_i \cap U_j .\)
      </center></i>
    </p>
    <p>
      <i>则</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \nu_p (f_i) = \nu_p \left( f_i (\varphi_{i j} (z_j))
        \frac{\mathrm{d}
\varphi_{i j} (z_j)}{\mathrm{d} z_j} \right) = \nu_p
        (f_i) .\)
      </center></i>
    </p>
    <p>
      <i>因而我们可以定义</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \nu_p (\omega) = \nu_p  (f_i), \quad p \in U_i .\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>如果 \(\nu_p (\omega) >
      0\)，则\(p\)称为是 \(\omega\)
      的零点；如果 \(\nu_p (\omega) <
      0\)，则\(p\)称为\(\omega\)的极点。</i>
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">4.5</class>.
      </strong><i>设\(C\)是一个Riemann面，</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \omega = \{ (U_i, z_i, f_i (z_i) \operatorname{dz}_i)
        \} \in K^1 (C),\)
      </center></i>
    </p>
    <p>
      <i>\(\gamma\)是\(C\)上分段光滑曲线，它不含\(\omega\)的极点，\(\gamma
      = \bigcup \gamma_i\) 是 \(\gamma\)
      的任意一个满足 \(\gamma_i
      \subset U_i\)
      的分拆。我们定义</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \int_{\gamma} \omega = \sum_i \int_{\gamma_i} f_i
        (z_i) \mathrm{d} z_i .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">4.6</class>.
      </strong><i>(全纯微分的Stokes定理)
      设\(C\)是一个Riemann面，\(\Omega
      \subset C\) 是一开集，\(\bar{\Omega}\)
      是紧致的，\(\partial \Omega = \gamma\)
      是一分段光滑曲线，\(\omega\)
      是在包含 \(\bar{\Omega}\)
      的一个开集上全纯的微分。则</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \int_{\partial \Omega} \omega = 0.\)
      </center></i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>例.
      </strong>证明Riemann球面\(S\)上除了平凡的全纯微分外没有其他全纯微分。
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">4.7</class>.
      </strong><i>\(\omega\) 在 \(p\)
      点的留数定义为</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \operatorname{Res}_p (\omega) = \frac{1}{2 \pi i}
        \oint_{\gamma_p} \omega .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>命题 <class style="font-style: normal">4.8</class>.
      </strong><i>(留数定理)
      设\(C\)是一个Riemann面，
      对于 \(\omega = \in K^1
      (C)\)，我们有</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \sum_{p \in C} \operatorname{Res}_p (\omega) = 0.\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">4.9</class>.
      </strong><i>如果 \(f \in K
      (C)\)不是常值函数，那么</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \sum_{p \in C} \nu_p (f) = 0.\)
      </center></i>
    </p>
    <p>
      <i>这意味着 \(f\)
      的零点个数等于 \(f\)
      的极点个数（重数计算在内）.</i>
    </p>
    <p style="margin-bottom: 1em">
      <i>证明：取 \(\omega = \mathrm{d} f /
      f\)并利用上面的留数定理。</i>
    </p>
    <p style="margin-top: 1em">
      <strong>推论 <class style="font-style: normal">4.10</class>.
      </strong><i>如果 \(f \in K (C)\)
      不是常值函数，那么对任意
      \(\alpha \in \mathbb{C}\) 都有</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \#f^{- 1} (\alpha) =\# \{ f的极点 \} .\)
      </center></i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">4.11</class>.
      </strong>如果把 \(f \in K (C)\)
      视为从 \(C\)
      到Riemann球面\(S\)的全纯映射，那么上面的推论可以解释为：\(S\)
      上的任意一个点，都是\(f\)的象，而且每个点的原香个数都相等。
    </p>
    <h2 id="auto-6">5<span style="margin-left: 1em"></span>微分形式<span style="margin-left: 1em"></span></h2>
    <p>
      一个复数 \(z = x + i y\)
      同时表示了一对实数。一个函数
      \(f : \mathbb{C} \rightarrow \mathbb{C}\)
      既可以视为 \(\mathbb{R}^2 \rightarrow
      \mathbb{R}^2\) 的映射
    </p>
    <center>
      \(\displaystyle \left\{ \begin{array}{l}
  u = u (x, y),\\
  v = v (x,
      y),
\end{array} \right.\)
    </center>
    <p>
      又可以视为\(\mathbb{R}^2 \rightarrow
      \mathbb{C}\) 的映射
    </p>
    <center>
      \(\displaystyle f = u (x, y) + i v (x, y) .\)
    </center>
    <p>
      引入一些记号。
    </p>
    <p style="margin-top: 1em">
      <strong>记法. </strong><i></i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \operatorname{dz}=\operatorname{dx}+
        i\operatorname{dy}, \quad \mathrm{d}
\bar{z} =\operatorname{dx}-
        i\operatorname{dy},\)
      </center></i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \frac{\partial f}{\partial z} = \frac{1}{2} \left(
        \frac{\partial f}{\partial
x} - i \frac{\partial f}{\partial y}
        \right)\)
      </center></i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \frac{\partial f}{\partial \bar{z}} = \frac{1}{2}
        \left( \frac{\partial
f}{\partial x} + i \frac{\partial f}{\partial y}
        \right),\)
      </center></i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \mathrm{d} f = \frac{\partial f}{\partial x}
        \mathrm{d} x + \frac{\partial
f}{\partial y} \mathrm{d} y =
        \frac{\partial f}{\partial z} \mathrm{d} z +
\frac{\partial
        f}{\partial \bar{z}} \mathrm{d} \bar{z}\)
      </center></i>
    </p>
    <p>
      <i>这时 \(\frac{\partial f}{\partial \bar{z}} = 0\)
      等价于Cauchy-Riemann方程</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \left\{ \begin{array}{l}
  \frac{\partial u}{\partial
        x} = \frac{\partial v}{\partial y},\\
  \frac{\partial u}{\partial y}
        = - \frac{\partial v}{\partial x},
\end{array} \right.\)
      </center></i>
    </p>
    <center>
      \(\displaystyle f = u (x, y) + i v (x, y) .\)
    </center>
    <p>
      因而 \(f\)
      是全纯的必须并且只须

    </p>
    <center>
      \(\displaystyle \frac{\partial f}{\partial \bar{z}} = 0.\)
    </center>
    <p>
      对于向量空间\(V\)中的元素，可以定义双线性、反对称的外积
      \(\wedge\):
    </p>
    <center>
      \(\displaystyle \left\{ \begin{array}{l}
  (a v_1) \wedge v_2 = a (v_1
      \wedge v_2)\\
  (v_1 + v_1') \wedge v_2 = v_1 \wedge v_2 + v_1' \wedge
      v_2\\
  v_2 \wedge v_1 = - v_1 \wedge v_2
\end{array} \right.\)
    </center>
    <p>
      采用上面的记号，实微分
      \(\operatorname{dx}, \operatorname{dy}\)
      的外积与复微分\(\operatorname{dz},
      \mathrm{d}
      \bar{z}\)的外积有以下关系：
    </p>
    <center>
      \(\displaystyle \operatorname{dz} \wedge \mathrm{d} \bar{z} = - 2
      i\operatorname{dx} \wedge
\operatorname{dy}.\)
    </center>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">5.1</class>.
      </strong><i>设\(C\)是一个Riemann面，\(\{
      (U_i, z_i) \}\)
      是它的全纯坐标覆盖。\(C\)
      上的一个微分1-形式
      \(\lambda\)，局部地由</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \lambda_i = f_i (z_i, \overline{z_i}) \mathrm{d} z_i +
        g_i (z_i, \bar{z}_i)
\mathrm{d} \bar{z}_i\)
      </center></i>
    </p>
    <p>
      <i>给出，这里 \(f_i, g_i\)
      是光滑函数，要求他们满足以下的变换法则</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle f_i (\varphi_{i j} (z_j), \overline{\varphi_{i j}
        (z_j)}) \frac{\mathrm{d}
\varphi_{i j} (z_j)}{\mathrm{d} z_j} = f_j
        (z_j, \bar{z}_j)\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle g_i (\varphi_{i j} (z_j), \overline{\varphi_{i j}
        (z_j)}) \frac{\mathrm{d}
\varphi_{i j} (z_j)}{\mathrm{d} z_j} = g_j
        (z_j, \bar{z}_j) .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">5.2</class>.
      </strong><i>微分1-形式 \(\lambda = \{ f_i
      \mathrm{d} z_i + g_i \mathrm{d} \overline{z_i} \}\)
      的外微分 \(\mathrm{d} \lambda\)
      定义为</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \begin{array}{rl}
  \mathrm{d} \lambda & = \{
        \mathrm{d} f_i \wedge \mathrm{d} z_i + \mathrm{d}
  g_i \wedge
        \mathrm{d} \bar{z}_i \}\\
  & = \left\{ \left( \frac{\partial
        g_i}{\partial z_i} - \frac{\partial
  f_i}{\partial \bar{z}_i} \right)
        \mathrm{d} z_i \wedge \mathrm{d} \bar{z}_i
  \right\} .
\end{array}\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">5.3</class>.
      </strong><i>\(C\)上的微分1-形式\(\lambda\)称为是闭1-形式，如果
      \(\mathrm{d} \lambda =
      0\);\(\lambda\)称为是恰当1-形式，如果存在光滑函数\(f\)，使得</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \lambda = \mathrm{d} f = \frac{\partial f}{\partial z}
        \mathrm{d} z +
\frac{\partial f}{\partial \bar{z}} \mathrm{d} \bar{z}
        .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>注意 <class style="font-style: normal">5.4</class>.
      </strong>恰当形式都是闭形式。
    </p>
    <center>
      \(\displaystyle \frac{\partial^2 f}{\partial z \partial \bar{z}} =
      \frac{\partial^2
f}{\partial \bar{z} \partial z}\)
    </center>
    <p style="margin-bottom: 1em">
      <center>
        \(\displaystyle \begin{array}{rl}
  \mathrm{d} (\mathrm{d} f) & =
        \mathrm{d} \left( \frac{\partial f}{\partial
  z} \mathrm{d} z +
        \frac{\partial f}{\partial \bar{z}} \mathrm{d} \bar{z}
  \right)\\
  &
        = \frac{\partial^2 f}{\partial \bar{z} \partial z} \mathrm{d} \bar{z}

        \wedge \mathrm{d} z + \frac{\partial^2 f}{\partial \bar{z} \partial
        z}
  \mathrm{d} z \wedge \mathrm{d} \bar{z}\\
  & = \left(
        \frac{\partial^2 f}{\partial z \partial \bar{z}} -
  \frac{\partial^2
        f}{\partial \bar{z} \partial z} \right) \mathrm{d} z \wedge

        \mathrm{d} \bar{z} = 0.
\end{array}\)
      </center>
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">5.5</class>.
      </strong><i>(微分形式的Stokes定理)
      设\(C\)是一Riemann面，\(\Omega\)
      是\(C\)上一个开集，\(\lambda\)
      是定义于包含 \(\bar{\Omega}\)
      的开集上的微分1-形式，则</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \int_{\partial \Omega} \lambda = \int \int_{\Omega}
        \mathrm{d} \lambda .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>推论 <class style="font-style: normal">5.6</class>.
      </strong><i>如果 \(\lambda\)
      是紧Riemann面\(C\)上的微分1-形式，那么</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \int \int_C \mathrm{d} \lambda = 0.\)
      </center></i>
    </p>
    <h2 id="auto-7">6<span style="margin-left: 1em"></span>Poincare-Hopf公式<span style="margin-left: 1em"></span></h2>
    <p>

    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">6.1</class>. </strong><i>设
      \(f : S^1 \rightarrow \mathbb{R}^2 \backslash \{ O
      \}\)是光滑映射，它的环绕数定义为</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \frac{1}{2 \pi} \oint_{S^1} f \ast \frac{u \mathrm{d}
        v - v \mathrm{d} u}{u^2
+ v^2} = \frac{1}{2 \pi} \int_{S^1} \mathrm{d}
        \arg f\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>定义 <class style="font-style: normal">6.2</class>.
      </strong><i>（微分形式的指标）设\(U\)是\(\mathbb{R}^2\)中包含原点\(O\)的一个开集，</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \lambda = a (x, y) \mathrm{d} x + b (x, y) \mathrm{d}
        y\)
      </center></i>
    </p>
    <p>
      <i>是定义于\(U\backslash \{ O
      \}\)的一个在各点均不为0的一个微分形式。考虑\(U\)中环绕原点\(O\)的任一小圆周\(S^1\)和映射</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \begin{array}{rl}
  g : & S^1 \rightarrow \mathbb{R}^2
        \backslash \{ O \}\\
  & p \mapsto (a (p), b (p)),
\end{array}\)
      </center></i>
    </p>
    <p>
      <i>微分形式\(\lambda\)在\(O\)点的指标就定义为映射\(g\)的环绕数</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \operatorname{Ind}_O \lambda
        \xequal{\operatorname{def}} \frac{1}{2 \pi}
\oint_{S^1} g \ast \frac{u
        \mathrm{d} v - v \mathrm{d} u}{u^2 + v^2} .\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">6.3</class>.
      </strong><i>(实微分形式的Poincare-Hopf指标公式)设\(C\)是可定向二维紧微分流形，\(\lambda\)是\(C\)上微分1-形式，它除了有限个孤立奇点外是光滑的，记这些奇点为\(p_1,
      \ldots, p_m .\) 则</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \sum_{i = 1}^m \operatorname{Ind}_{p_i} \lambda
        =\boldsymbol{\chi} (C) .\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>这里 \(\boldsymbol{\chi} (C) = 2 - 2 g\)
      是\(C\)的Euler示性数。</i>
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">6.4</class>.
      </strong>通常对于向量场陈述的Poincare-Hopf指标公式与这里的形式等价。
    </p>
    <p style="margin-top: 1em">
      <strong>定理 <class style="font-style: normal">6.5</class>.
      </strong><i>（关于半纯微分的Poincare-Hopf指标公式）设\(C\)是紧Riemann面，\(\omega
      \in K^1 (C)\)，则</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \sum_{p \in C} \nu_p (\omega) = -\boldsymbol{\chi} (C)
        .\)
      </center></i>
    </p>
    <p>
      后面内容是正则化、Riemann-Roch定理及应用，Abel定理及应用。
    </p>
    <p>

    </p>
  </body>

<!-- prettier-ignore-end -->
