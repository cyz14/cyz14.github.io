---
layout: post
title: Hessian
date: 2026-11-25 20:50:00 +0800
description: Lectures of Differential Geometry, chapter 1
tags: connection, curvature, Hessian
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
            <td><font style="font-size: 168.2%"><strong>&#x6D41;&#x5F62;&#x4E0A;&#x4EFF;&#x5C04;&#x8054;&#x7EDC;&#x3001;Hessian&#x3001;Laplacian&#x7684;&#x5B9A;&#x4E49;</strong></font></td>
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
                        <class style="font-variant: small-caps">by
                        &#x300A;&#x5FAE;&#x5206;&#x51E0;&#x4F55;&#x8BB2;&#x4E49;&#x300B;&#xFF0C;DeepSeek&#xFF0C;GPT</class>
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
    <p>

    </p>
    <h2 id="auto-1">1<span style="margin-left: 1em"></span>&#x4EFF;&#x5C04;&#x8054;&#x7EDC;<span style="margin-left: 1em"></span></h2>
    <h3 id="auto-2">1.1<span style="margin-left: 1em"></span>&#x4EFF;&#x5C04;&#x8054;&#x7EDC;&#x7684;&#x5B9A;&#x4E49;<span
    style="margin-left: 1em"></span></h3>
    <p>
      &#x8BBE;\(M\)&#x662F;&#x5149;&#x6ED1;&#x6D41;&#x5F62;&#xFF0C;\(T
      M\)&#x662F;&#x5207;&#x4E1B;&#xFF0C;\(\Gamma (T
      M)\)&#x8868;&#x793A;&#x5149;&#x6ED1;&#x5411;&#x91CF;&#x573A;&#x7A7A;&#x95F4;&#x3002;
    </p>
    <p>
      &#x4E00;&#x4E2A;&#x4EFF;&#x5C04;&#x8054;&#x7EDC;&#x662F;&#x4E00;&#x4E2A;&#x6620;&#x5C04;
    </p>
    <center>
      \(\displaystyle \nabla : \Gamma (T M) \times \Gamma (T M) \rightarrow
      \Gamma (T M), \quad (X,
Y) \mapsto \nabla_X Y\)
    </center>
    <p>
      &#x6EE1;&#x8DB3;&#x4E24;&#x6761;&#x516C;&#x7406;&#xFF1A;
    </p>
    <p>
      (A1) &#x5BF9;&#x7B2C;&#x4E00;&#x4E2A;&#x53D8;&#x91CF;\(C^{\infty}
      (M)\)-&#x7EBF;&#x6027;
    </p>
    <center>
      \(\displaystyle \nabla_{f X + g Y} Z = f \nabla_X Z + g \nabla_Y Z.\)
    </center>
    <p>
      (A2) &#x5BF9;&#x7B2C;&#x4E8C;&#x4E2A;&#x53D8;&#x91CF;&#x6EE1;&#x8DB3;
      Leibniz &#x5F8B;
    </p>
    <center>
      \(\displaystyle \nabla_X (f Y) = X (f) Y + f \nabla_X Y.\)
    </center>
    <p>
      &#x8FD9;&#x6761;&#x8BF4;&#x660E;&#xFF1A;\(\nabla_X\)&#x4F5C;&#x7528;&#x5728;&#x5411;&#x91CF;&#x573A;&#x4E0A;&#x50CF;&#x4E00;&#x4E2A;&ldquo;&#x6CBF;\(X\)&#x65B9;&#x5411;&#x7684;&#x5BFC;&#x6570;&rdquo;&#x3002;
    </p>
    <h3 id="auto-3">1.2<span style="margin-left: 1em"></span>Leibniz&#x5F8B;&#xFF1A;&#x4ECE;&#x5411;&#x91CF;&#x573A;&#x6269;&#x5C55;&#x5230;&#x5F20;&#x91CF;&#x573A;<span
    style="margin-left: 1em"></span></h3>
    <p>
      Leibniz&#x5F8B;&#x4E0D;&#x53EA;&#x662F;&#x5BF9; \(f Y\)
      &#x8FD9;&#x4E00;&#x6761;&#x3002;&#x5B83;&#x662F;&#x628A;\(\nabla\)&#x4ECE;&#x5411;&#x91CF;&#x573A;&#x552F;&#x4E00;&#x6269;&#x5C55;&#x5230;&#x6240;&#x6709;&#x5F20;&#x91CF;&#x573A;&#x7684;&#x6838;&#x5FC3;&#x539F;&#x5219;&#x3002;
    </p>
    <h4 id="auto-4">1.2.1<span style="margin-left: 1em"></span>&#x5BF9;&#x51FD;&#x6570;<span style="margin-left: 1em"></span></h4>
    <p>
      &#x89C4;&#x5B9A;
    </p>
    <center>
      \(\displaystyle \nabla_X f = X (f) .\)
    </center>
    <h4 id="auto-5">1.2.2<span style="margin-left: 1em"></span>&#x5BF9;\(1\)-&#x5F62;&#x5F0F;<span
    style="margin-left: 1em"></span></h4>
    <p>
      &#x8981;&#x6C42;\(\nabla\)&#x4E0E;&#x7F29;&#x5E76;&#x76F8;&#x5BB9;&#xFF0C;&#x5373;&#x5BF9;\(1\)-&#x5F62;&#x5F0F;\(\omega\)&#x548C;&#x5411;&#x91CF;&#x573A;\(Y\)&#xFF0C;
    </p>
    <center>
      \(\displaystyle \nabla_X (\omega (Y)) = (\nabla_X \omega) (Y) + \omega
      (\nabla_X Y) .\)
    </center>
    <p>
      &#x5DE6;&#x8FB9;&#x662F;&#x51FD;&#x6570;&#x7684;&#x65B9;&#x5411;&#x5BFC;&#x6570;\(X
      (\omega (Y))\)&#x3002;&#x79FB;&#x9879;&#x5F97;
    </p>
    <center>
      \(\displaystyle (\nabla_X \omega) (Y) = X (\omega (Y)) - \omega
      (\nabla_X Y) .\)
    </center>
    <p>
      &#x8FD9;&#x91CC;&#x7684;&#x8D1F;&#x53F7;&#x662F;Leibniz&#x5F8B;&#x7684;&#x4EA7;&#x7269;&#x3002;
    </p>
    <h4 id="auto-6">1.2.3<span style="margin-left: 1em"></span>&#x5BF9;&#x5F20;&#x91CF;&#x79EF;<span
    style="margin-left: 1em"></span></h4>
    <p>
      &#x8981;&#x6C42;
    </p>
    <center>
      \(\displaystyle \nabla_X (T \otimes S) = (\nabla_X T) \otimes S + T
      \otimes (\nabla_X S) .\)
    </center>
    <h4 id="auto-7">1.2.4<span style="margin-left: 1em"></span>&#x5BF9;&#x4E00;&#x822C;\((r,
    s)\)&#x578B;&#x5F20;&#x91CF;<span style="margin-left: 1em"></span></h4>
    <p>
      &#x7EFC;&#x5408;&#x4EE5;&#x4E0A;&#xFF0C;&#x5BF9;\(T \in \Gamma (T^{r, s}
      M)\)&#xFF0C;
    </p>
    <center>
      \(\displaystyle (\nabla_X T) (\omega^1, \ldots, \omega^r, Y_1, \ldots,
      Y_s) = X (T (\omega^1,
\ldots, Y_s)) - \sum_a T (\ldots, \nabla_X
      \omega^a, \ldots, Y_s) - \sum_b T
(\ldots, \omega^a, \ldots, \nabla_X
      Y_b, \ldots) .\)
    </center>
    <p>
      &#x534F;&#x53D8;&#x6307;&#x6807;
      \((Y_b)\)&#x5E26;&#x8D1F;&#x53F7;&#xFF0C;&#x9006;&#x53D8;&#x6307;&#x6807;
      \((\omega^a)\)
      &#x4E5F;&#x5E26;&#x8D1F;&#x53F7;&#xFF0C;&#x56E0;&#x4E3A;&#x7F29;&#x5E76;&#x65F6;&#x8981;&#x6263;&#x6389;&ldquo;&#x6807;&#x67B6;&#x53D8;&#x5316;&rdquo;&#x7684;&#x90E8;&#x5206;&#x3002;
    </p>
    <p>
      &#x524D;&#x9762;Hessian&#x7684;&#x534F;&#x53D8;&#x5BFC;&#x6570;
    </p>
    <center>
      \(\displaystyle (\nabla_X H) (Y, Z) = X (H (Y, Z)) - H (\nabla_X Y, Z) -
      H (Y, \nabla_X Z)\)
    </center>
    <p>
      &#x6B63;&#x662F;&#x8FD9;&#x4E2A;&#x516C;&#x5F0F;&#x5728;\((0,
      2)\)&#x578B;&#x5F20;&#x91CF;&#x4E0A;&#x7684;&#x7279;&#x4F8B;&#x3002;
    </p>
    <h3 id="auto-8">1.3<span style="margin-left: 1em"></span>&#x5EA6;&#x91CF;&#x76F8;&#x5BB9;&#x6027;<span
    style="margin-left: 1em"></span></h3>
    <p>
      &#x8BBE;\((M,
      g)\)&#x662F;&#x9ECE;&#x66FC;&#x6D41;&#x5F62;&#xFF0C;\(\nabla\)&#x662F;&#x4EFF;&#x5C04;&#x8054;&#x7EDC;&#x3002;&#x79F0;\(\nabla\)&#x4E0E;&#x5EA6;&#x91CF;&#x76F8;&#x5BB9;&#xFF0C;&#x82E5;
    </p>
    <center>
      \(\displaystyle \nabla g = 0,\)
    </center>
    <p>
      &#x7B49;&#x4EF7;&#x5730;&#xFF0C;&#x5BF9;&#x4EFB;&#x610F;&#x5411;&#x91CF;&#x573A;
      \(X, Y, Z\)&#xFF0C;
    </p>
    <center>
      \(\displaystyle X (g (Y, Z)) = g (\nabla_X Y, Z) + g (Y, \nabla_X Z) .\)
    </center>
    <p>
      &#x76F4;&#x89C2;&#x610F;&#x4E49;
    </p>
    <ul>
      <li>
        <p>
          &#x6CBF;&#x4EFB;&#x610F;&#x66F2;&#x7EBF;&#x5E73;&#x884C;&#x79FB;&#x52A8;&#x5411;&#x91CF;&#x65F6;&#xFF0C;&#x5185;&#x79EF;&#x4FDD;&#x6301;&#x4E0D;&#x53D8;&#xFF1B;
        </p>
      </li>
      <li>
        <p>
          &#x5E73;&#x884C;&#x79FB;&#x52A8;&#x662F;&#x6B63;&#x4EA4;&#x53D8;&#x6362;&#xFF1B;
        </p>
      </li>
      <li>
        <p>
          &#x53EF;&#x4EE5;&#x7528;\(\nabla\)&#x548C;\(g\)&#x81EA;&#x7531;&#x5730;&#x964D;&#x6307;&#x6807;&#x3001;&#x5347;&#x6307;&#x6807;&#xFF0C;&#x56E0;&#x4E3A;\(g\)&#x662F;&#x5E73;&#x884C;&#x7684;&#x3002;
        </p>
      </li>
    </ul>
    <p>
      &#x5C40;&#x90E8;&#x5750;&#x6807;&#x5F62;&#x5F0F;
    </p>
    <center>
      \(\displaystyle \partial_k g_{i j} = \Gamma_{k i}^l g_{l j} +
      \Gamma^l_{k j} g_{i l} .\)
    </center>
    <h3 id="auto-9">1.4<span style="margin-left: 1em"></span>&#x65E0;&#x6320;&#x6027;<span style="margin-left: 1em"></span></h3>
    <p>
      &#x4EFF;&#x5C04;&#x8054;&#x7EDC;&#x7684;&#x6320;&#x7387;&#x5F20;&#x91CF;&#x5B9A;&#x4E49;&#x4E3A;
    </p>
    <center>
      \(\displaystyle T (X, Y) = \nabla_X Y - \nabla_Y X - [X, Y] .\)
    </center>
    <p>
      &#x79F0;\(\nabla\)&#x65E0;&#x6320;&#xFF0C;&#x82E5;
    </p>
    <center>
      \(\displaystyle \nabla_X Y - \nabla_Y X = [X, Y] .\)
    </center>
    <p>
      &#x5C40;&#x90E8;&#x5750;&#x6807;&#x4E2D;&#xFF0C;&#x65E0;&#x6320;&#x7B49;&#x4EF7;&#x4E8E;Christoffel&#x7B26;&#x53F7;&#x5BF9;&#x79F0;&#xFF1A;
    </p>
    <center>
      \(\displaystyle \Gamma_{i j}^k = \Gamma_{j i}^k .\)
    </center>
    <h3 id="auto-10">1.5<span style="margin-left: 1em"></span>Levi-Civita
    &#x8054;&#x7EDC;&#xFF1A;&#x57FA;&#x672C;&#x5B9A;&#x7406;<span style="margin-left: 1em"></span></h3>
    <p>
      &#x9ECE;&#x66FC;&#x51E0;&#x4F55;&#x57FA;&#x672C;&#x5B9A;&#x7406;&#xFF1A;&#x5728;&#x9ECE;&#x66FC;&#x6D41;&#x5F62;\((M,
      g)\)&#x4E0A;&#xFF0C;&#x5B58;&#x5728;&#x552F;&#x4E00;&#x7684;&#x4EFF;&#x5C04;&#x8054;&#x7EDC;\(\nabla\)&#xFF0C;&#x540C;&#x65F6;&#x6EE1;&#x8DB3;:
    </p>
    <ol>
      <li>
        <p>
          &#x65E0;&#x6320;&#xFF1A;\(\nabla_X Y - \nabla_Y X = [X, Y]\);
        </p>
      </li>
      <li>
        <p>
          &#x5EA6;&#x91CF;&#x76F8;&#x5BB9;&#xFF1A;\(\nabla g = 0\).
        </p>
      </li>
    </ol>
    <p>
      &#x8FD9;&#x4E2A;&#x8054;&#x7EDC;&#x79F0;&#x4E3A;Levi-Civita&#x8054;&#x7EDC;&#xFF08;&#x6216;&#x9ECE;&#x66FC;&#x8054;&#x7EDC;&#xFF09;&#x3002;
    </p>
    <p>
      Koszul&#x516C;&#x5F0F;
    </p>
    <p>
      &#x5B83;&#x7531;&#x5EA6;&#x91CF;&#x552F;&#x4E00;&#x786E;&#x5B9A;&#xFF1A;
    </p>
    <center>
      \(\displaystyle 2 g (\nabla_X Y, Z) = X (g (Y, Z)) + Y (g (X, Z)) - Z (g
      (X, Y)) + g ([X, Y],
Z) - g ([X, Z], Y) - g\)
    </center>
    <p>
      &#x5C40;&#x90E8;&#x5750;&#x6807;&#xFF1A;Christoffel&#x7B26;&#x53F7;
    </p>
    <center>
      \(\displaystyle \nabla_{\partial_i} \partial_j = \Gamma_{i j}^k \partial
      k\)
    </center>
    <center>
      \(\displaystyle \Gamma_{i j}^k = \frac{1}{2} g^{k l} (\partial_i g_{j l}
      + \partial_j g_{i l}
- \partial_l g_{i j}) .\)
    </center>
    <h3 id="auto-11">1.6<span style="margin-left: 1em"></span>&#x5E94;&#x7528;<span style="margin-left: 1em"></span></h3>
    <h4 id="auto-12">1.6.1<span style="margin-left: 1em"></span>Hessian&#x7684;&#x5BF9;&#x79F0;&#x6027;&#x6765;&#x81EA;&#x65E0;&#x6320;<span
    style="margin-left: 1em"></span></h4>
    <center>
      \(\displaystyle (\operatorname{Hess}f) (X, Y) = (\nabla_X d f) (Y) = X
      (Y f) - (\nabla_X Y) f.\)
    </center>
    <p>
      &#x4EA4;&#x6362;\(X, Y\)&#x76F8;&#x51CF;&#xFF1A;
    </p>
    <center>
      \(\displaystyle (\operatorname{Hess}f) (X, Y) - (\operatorname{Hess}f)
      (Y, X) = [X, Y] f -
(\nabla_X Y - \nabla_Y X) f.\)
    </center>
    <p>
      &#x65E0;&#x6320;&#x65F6;\(\nabla_X Y - \nabla_Y X = [X,
      Y]\)&#xFF0C;&#x53F3;&#x8FB9;&#x4E3A;0&#x3002;&#x6240;&#x4EE5;
    </p>
    <center>
      \(\displaystyle \operatorname{Hess}f (X, Y) =\operatorname{Hess}f (Y, X)
      .\)
    </center>
    <h4 id="auto-13">1.6.2<span style="margin-left: 1em"></span>Hessian&#x5199;&#x6210;&#x5185;&#x79EF;&#x5F62;&#x5F0F;&#x4F9D;&#x8D56;&#x5EA6;&#x91CF;&#x76F8;&#x5BB9;<span
    style="margin-left: 1em"></span></h4>
    <center>
      \(\displaystyle \operatorname{Hess}f (X, Y) = g (\nabla_X \nabla f, Y)\)
    </center>
    <p>
      &#x4E2D;&#xFF0C;\(\nabla f\) &#x7531; \(g (\nabla f, X) = d f
      (X)\)&#x5B9A;&#x4E49;&#xFF0C;&#x5185;&#x79EF;&#x662F;\(g\)&#x3002;&#x5EA6;&#x91CF;&#x76F8;&#x5BB9;&#x4FDD;&#x8BC1;&#x8FD9;&#x5957;&ldquo;&#x5347;&#x6307;&#x6807;-&#x6C42;&#x5BFC;-&#x964D;&#x6307;&#x6807;&rdquo;&#x7684;&#x64CD;&#x4F5C;&#x4E0E;
      \(\nabla d f\) &#x4E00;&#x81F4;&#x3002;
    </p>
    <h4 id="auto-14">1.6.3<span style="margin-left: 1em"></span>&#x8054;&#x7EDC;Laplacian&#x7684;&#x4F34;&#x968F;&#x4F9D;&#x8D56;&#x5EA6;&#x91CF;<span
    style="margin-left: 1em"></span></h4>
    <center>
      \(\displaystyle L = - \nabla^{\ast} \nabla\)
    </center>
    <p>
      &#x4E2D; \(\nabla^{\ast}\) &#x662F; \(\nabla\) &#x5173;&#x4E8E; \(L^2\)
      &#x5185;&#x79EF;&#x7684;&#x6B63;&#x5F0F;&#x4F34;&#x968F;&#x3002;\(L^2\)
      &#x5185;&#x79EF;&#x7531; \(g\)
      &#x548C;&#x4F53;&#x79EF;&#x5143;&#x5B9A;&#x4E49;&#xFF0C;&#x6240;&#x4EE5;
      \(L\)
      &#x7684;&#x826F;&#x5B9A;&#x4E49;&#x4F9D;&#x8D56;&#x5EA6;&#x91CF;&#x76F8;&#x5BB9;&#x6027;&#x3002;
    </p>
    <h4 id="auto-15">1.6.4<span style="margin-left: 1em"></span>&#x6B63;&#x4EA4;&#x6807;&#x67B6;&#x548C;&#x8FF9;<span
    style="margin-left: 1em"></span></h4>
    <center>
      \(\displaystyle \langle L \nabla f, e_j \rangle = \sum_i (\nabla_{e_i}
      H) (e_j, e_i)\)
    </center>
    <p>
      &#x4E2D;&#xFF0C;\(\{ e_i \}\)&#x662F;
      \(g\)-&#x6B63;&#x4EA4;&#x6807;&#x67B6;&#xFF0C;\(\sum_i\)&#x662F;&#x53D6;&#x8FF9;&#x3002;&#x8FD9;&#x4E9B;&#x90FD;&#x4F9D;&#x8D56;
      \(g\) &#x548C; \(\nabla g = 0\)&#x3002;
    </p>
    <h2 id="auto-16">2<span style="margin-left: 1em"></span>Hessian<span style="margin-left: 1em"></span></h2>
    <h3 id="auto-17">2.1<span style="margin-left: 1em"></span>&#x9884;&#x5907;&#x8BB0;&#x53F7;<span
    style="margin-left: 1em"></span></h3>
    <p>
      &#x5207;&#x4E1B; \(T M\)&#xFF0C;&#x4F59;&#x5207;&#x4E1B; \(T^{\ast}
      M\)&#x3002;
    </p>
    <p>
      &#x5411;&#x91CF;&#x573A;&#x7A7A;&#x95F4; \(\Gamma (T
      M)\)&#xFF0C;1-&#x5F62;&#x5F0F;&#x7A7A;&#x95F4; \(\Gamma (T^{\ast}
      M)\)&#x3002;
    </p>
    <p>
      &#x5EA6;&#x91CF; \(g (\cdot, \cdot) = \langle \cdot, \cdot
      \rangle\)&#x3002;
    </p>
    <p>
      &#x68AF;&#x5EA6; \(\nabla f =\operatorname{grad}f\) &#x7531;
    </p>
    <center>
      \(\displaystyle g (\nabla f, X) = d f (X) = X f\)
    </center>
    <p>
      &#x552F;&#x4E00;&#x786E;&#x5B9A;&#x3002;
    </p>
    <p>
      &#x534F;&#x53D8;&#x5BFC;&#x6570; \(\nabla_X Y\) &#x5BF9; \(X\) &#x662F;
      \(C^{\infty} (M)\)-&#x7EBF;&#x6027;&#x7684;&#xFF0C;&#x5BF9; \(Y\)
      &#x6EE1;&#x8DB3;Leibniz&#x5F8B;&#x3002;
    </p>
    <h3 id="auto-18">2.2<span style="margin-left: 1em"></span>Hessian&#x7684;&#x5B9A;&#x4E49;<span style="margin-left: 1em"></span></h3>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49; <class style="font-style: normal">1</class>.
      </strong><i>(&#x534F;&#x53D8;&#x5BFC;&#x6570;&#x5B9A;&#x4E49;)</i>
    </p>
    <p>
      <i>&#x51FD;&#x6570;\(f \in C^{\infty}
      (M)\)&#x7684;Hessian&#x5B9A;&#x4E49;&#x4E3A; \(d f\)
      &#x7684;&#x534F;&#x53D8;&#x5BFC;&#x6570;</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \operatorname{Hess}f := \nabla d f \in \Gamma
        (T^{\ast} M \otimes T^{\ast} M)
.\)
      </center></i>
    </p>
    <p>
      <i>&#x5B83;&#x5BF9; \(X, Y \in \Gamma (T
      M)\)&#x7684;&#x4F5C;&#x7528;&#x4E3A;</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle (\operatorname{Hess}f) (X, Y) = (\nabla_X d f) (Y) = X
        (Y f) - (\nabla_X Y) f.\)
      </center></i>
    </p>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49; <class style="font-style: normal">2</class>.
      </strong><i>&#xFF08;&#x68AF;&#x5EA6;-&#x5185;&#x79EF;&#x5B9A;&#x4E49;&#xFF09;</i>
    </p>
    <p>
      <i>&#x7B49;&#x4EF7;&#x5730;&#xFF0C;</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle (\operatorname{Hess}f) (X, Y) = g (\nabla_X \nabla f,
        Y) .\)
      </center></i>
    </p>
    <p>
      <i>&#x8FD9;&#x4E0E;&#x5B9A;&#x4E49;1&#x4E00;&#x81F4;&#xFF0C;&#x56E0;&#x4E3A;</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle g (\nabla_X \nabla f, Y) = X g (\nabla f, Y) - g
        (\nabla f, \nabla_X Y) = X (Y
f) - (\nabla_X Y) f.\)
      </center></i>
    </p>
    <p>
      &#x6027;&#x8D28;
    </p>
    <ol>
      <li>
        <p>
          &#x5BF9;&#x79F0;&#x6027;&#xFF08;&#x7531;\(\nabla\)&#x65E0;&#x6320;&#xFF09;
        </p>
        <center>
          \(\displaystyle (\operatorname{Hess}f) (X, Y) =
          (\operatorname{Hess}f) (Y, X) .\)
        </center>
      </li>
      <li>
        <p>
          &#x5F20;&#x91CF;&#x6027;&#xFF1A;\(\operatorname{Hess}f\) &#x662F;
          \((0, 2)\)&#x578B;&#x5F20;&#x91CF;&#xFF0C;&#x5BF9; \(X, Y\) &#x5747;
          \(C^{\infty} (M)\)-&#x7EBF;&#x6027;&#x3002;
        </p>
      </li>
      <li>
        <p>
          &#x5C40;&#x90E8;&#x5750;&#x6807;&#xFF1A;
        </p>
        <center>
          \(\displaystyle (\operatorname{Hess}f)_{i j} = \partial_i \partial_j
          f - \Gamma_{i j}^k
\partial_k f.\)
        </center>
        <p>
          &#x5728;&#x4E34;&#x754C;&#x70B9; \(d f = 0\)
          &#x5904;&#x9000;&#x5316;&#x4E3A;&#x666E;&#x901A;&#x4E8C;&#x9636;&#x504F;&#x5BFC;&#x77E9;&#x9635;&#x3002;
        </p>
      </li>
      <li>
        <p>
          &#x4E0E;&#x5EA6;&#x91CF;&#x7684;&#x5173;&#x7CFB;&#xFF1A;\(\operatorname{Hess}f\)
          &#x4E0E; \(g\)
          &#x540C;&#x578B;&#xFF0C;&#x53EF;&#x5B9A;&#x4E49;&#x5176;
          \(g\)-&#x8FF9;&#xFF0C;&#x5373;Laplace&#x7B97;&#x5B50;&#x3002;
        </p>
      </li>
    </ol>
    <h3 id="auto-19">2.3<span style="margin-left: 1em"></span>\(\nabla H\): Hessian
    &#x7684;&#x534F;&#x53D8;&#x5BFC;&#x6570;<span style="margin-left: 1em"></span></h3>
    <p>
      &#x4EE4;
    </p>
    <center>
      \(\displaystyle H =\operatorname{Hess}f = \nabla d f.\)
    </center>
    <p>
      &#x5B83;&#x662F;&#x4E00;&#x4E2A; \((0,
      2)\)-&#x578B;&#x5F20;&#x91CF;&#x3002;&#x5BF9;&#x5B83;&#x518D;&#x6C42;&#x534F;&#x53D8;&#x5BFC;&#x6570;&#xFF0C;&#x5F97;&#x5230;
      \((0, 3)\)-&#x578B;&#x5F20;&#x91CF;
    </p>
    <center>
      \(\displaystyle \nabla H \in \Gamma (T^{\ast} M \otimes T^{\ast} M
      \otimes T^{\ast} M) .\)
    </center>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49;. </strong><i>(Leibniz &#x6269;&#x5C55;)</i>
    </p>
    <p>
      <i>&#x5BF9; \(X, Y, Z \in \Gamma (T M)\),</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle (\nabla_X H) (Y, Z) = X (H (Y, Z)) - H (\nabla_X Y, Z)
        - H (Y, \nabla_X Z) .\)
      </center></i>
    </p>
    <p>
      <i>&#x53D6; \(X = e_i\):</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle (\nabla_{e_i} H) (Y, Z) = e_i (H (Y, Z)) - H
        (\nabla_{e_i} Y, Z) - H (Y,
\nabla_{e_i} Z) .\)
      </center></i>
    </p>
    <p>
      &#x5C40;&#x90E8;&#x5750;&#x6807;
    </p>
    <center>
      \(\displaystyle (\nabla_i H)_{j k} = \partial_i H_{j k} - \Gamma_{i j}^l
      H_{l k} - \Gamma_{i
k}^l H_{j l} .\)
    </center>
    <p>
      &#x6027;&#x8D28;
    </p>
    <ol>
      <li>
        <p>
          &#x5F20;&#x91CF;&#x6027;&#xFF1A;\(\nabla H\)&#x5BF9; \(X, Y, Z\)
          &#x5747; \(C^{\infty} (M)\)-&#x7EBF;&#x6027;
        </p>
      </li>
      <li>
        <p>
          &#x5BF9;&#x79F0;&#x6027;&#xFF1A;&#x7531;&#x4E8E; \(H\)
          &#x5BF9;&#x79F0;&#x4E14; \(\nabla\)&#x65E0;&#x6320;&#xFF0C;\(\nabla
          H\)
          &#x5728;&#x540E;&#x4E24;&#x4E2A;&#x6307;&#x6807;&#x4E0A;&#x5BF9;&#x79F0;&#xFF1A;
        </p>
        <center>
          \(\displaystyle (\nabla_X H) (Y, Z) = (\nabla_X H) (Z, Y) .\)
        </center>
      </li>
      <li>
        <p>
          \(\nabla H\) &#x662F; \(\operatorname{Hess}f\)
          &#x7684;&#x4E09;&#x9636;&#x5BFC;&#x6570;&#xFF0C;&#x5728;&#x4E34;&#x754C;&#x70B9;&#x9644;&#x8FD1;&#x523B;&#x753B;
          Hessian &#x7684;&#x53D8;&#x5316;&#x7387;&#x3002;
        </p>
      </li>
    </ol>
    <h3 id="auto-20">2.4<span style="margin-left: 1em"></span>Laplace
    &#x7B97;&#x5B50;&#x7684;&#x5B9A;&#x4E49;<span style="margin-left: 1em"></span></h3>
    <p>
      &#x6D41;&#x5F62;&#x4E0A;&#x6709;&#x4E24;&#x4E2A;&#x5E38;&#x7528;Laplace&#x7B97;&#x5B50;
    </p>
    <h4 id="auto-21">2.4.1<span style="margin-left: 1em"></span>Laplace-Beltrami
    &#x7B97;&#x5B50;&#xFF0C;&#x4F5C;&#x7528;&#x5728;&#x51FD;&#x6570;&#x4E0A;<span
    style="margin-left: 1em"></span></h4>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49;.
      </strong><i>(&#x6563;&#x5EA6;-&#x68AF;&#x5EA6;&#xFF09;</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \Delta f =\operatorname{div} (\nabla f) .\)
      </center></i>
    </p>
    <p>
      <i>&#x5176;&#x4E2D;&#x6563;&#x5EA6;&#x7531;</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \operatorname{div}X =\operatorname{tr}_g (\nabla X) =
        \sum_i g (\nabla_{e_i}
X, e_i)\)
      </center></i>
    </p>
    <p style="margin-bottom: 1em">
      <i>&#x5B9A;&#x4E49;&#x3002;</i>
    </p>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49;. </strong><i>(Hessian &#x7684;&#x8FF9;)</i>
    </p>
    <p>
      <i>&#x7B49;&#x4EF7;&#x5730;</i>
    </p>
    <p>
      <i><center>
        \(\displaystyle \Delta f =\operatorname{tr}_g (\operatorname{Hess}f) =
        \sum_i
(\operatorname{Hess}f) (e_i, e_i) = \sum_i g (\nabla_{e_i}
        \nabla f, e_i) .\)
      </center></i>
    </p>
    <p>
      <i>&#x5C40;&#x90E8;&#x5750;&#x6807;</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \Delta f = \frac{1}{\sqrt{| g |}} \partial_i \left(
        \sqrt{| g |} g^{i j}
\partial_j f \right) .\)
      </center></i>
    </p>
    <p>
      &#x6027;&#x8D28;
    </p>
    <ul>
      <li>
        <p>
          \(\Delta f\) &#x662F;&#x6807;&#x91CF;&#x51FD;&#x6570;&#x3002;
        </p>
      </li>
      <li>
        <p>
          &#x7B26;&#x53F7;&#x7EA6;&#x5B9A;&#xFF1A;&#x51E0;&#x4F55;&#x4E2D;&#x5E38;&#x7528;
          \(\Delta =\operatorname{div}
          \nabla\)(&#x975E;&#x8D1F;&#x8C31;&#x7684;&#x53CD;&#x53F7;&#x53D6;&#x51B3;&#x4E8E;&#x7EA6;&#x5B9A;)&#xFF0C;&#x5206;&#x6790;&#x4E2D;&#x5E38;&#x7528;\(-
          \Delta\)&#x4F7F;&#x5176;&#x6B63;&#x5B9A;&#x3002;
        </p>
      </li>
      <li>
        <p>
          &#x5728; \(\mathbb{R}^n\)
          &#x4E2D;&#x9000;&#x5316;&#x4E3A;&#x666E;&#x901A; Laplacian \(\sum_i
          \partial_i^2 f\).
        </p>
      </li>
    </ul>
    <h4 id="auto-22">2.4.2<span style="margin-left: 1em"></span>&#x8054;&#x7EDC;Laplacian &#xFF08;Bochner
    Laplacian&#xFF0C;&#x4F5C;&#x7528;&#x5728;&#x5F20;&#x91CF;&#x573A;/&#x5411;&#x91CF;&#x573A;&#x4E0A;&#xFF09;<span
    style="margin-left: 1em"></span></h4>
    <p>
      &#x5BF9;&#x5411;&#x91CF;&#x573A; \(X\),
    </p>
    <center>
      \(\displaystyle L X = - \nabla^{\ast} \nabla X = \sum_i \left(
      \nabla_{e_i} \nabla_{e_i} X -
\nabla_{\nabla_{e_i} e_i} X \right) .\)
    </center>
    <p>
      &#x7B26;&#x53F7;&#x7EA6;&#x5B9A;&#x4E0D;&#x540C;&#x5219;&#x6574;&#x4F53;&#x5DEE;&#x8D1F;&#x53F7;&#x3002;
    </p>
    <p>
      &#x5BF9;1-&#x5F62;&#x5F0F; \(\omega\),&#x540C;&#x7406;
    </p>
    <center>
      \(\displaystyle L \omega = - \nabla^{\ast} \nabla \omega,\)
    </center>
    <p>
      &#x5C40;&#x90E8;&#x4E0A;
    </p>
    <center>
      \(\displaystyle (L \omega)_j = - g^{i k} (\nabla_i \nabla_k \omega)_j
      .\)
    </center>
    <p>
      &#x4E0E;&#x51FD;&#x6570;Laplacian&#x7684;&#x5173;&#x7CFB;
    </p>
    <p>
      &#x5BF9;&#x51FD;&#x6570; \(f\)&#xFF0C;
    </p>
    <center>
      \(\displaystyle L f {=\!\!=} - \nabla^{\ast} \nabla f = - \Delta f\)
    </center>
    <p>
      &#xFF08;&#x5728; \(\Delta =\operatorname{div}
      \nabla\)&#x7EA6;&#x5B9A;&#x4E0B;&#x5DEE;&#x8D1F;&#x53F7;&#xFF09;&#x3002;
    </p>
    <p>
      &#x4F46;&#x5BF9;1-&#x5F62;&#x5F0F;\(d f\)&#xFF0C;\(L (d f)\)&#x4E0E; \(d
      (\Delta f)\)
      &#x4E00;&#x822C;&#x4E0D;&#x540C;&#xFF0C;&#x4E8C;&#x8005;&#x7531;
      Weitzenbock &#x516C;&#x5F0F;&#x8054;&#x7CFB;
    </p>
    <center>
      \(\displaystyle \Delta_H d f = d (\Delta f) +\operatorname{Ric} (\nabla
      f),\)
    </center>
    <p>
      &#x5176;&#x4E2D; \(\Delta_H\) &#x662F; Hodge-de Rham
      Laplacian&#x3002;&#x8054;&#x7EDC; Laplacian \(L\) &#x4E0E; Hodge
      Laplacian&#x4E5F;&#x5DEE; Ricci&#x9879;&#x3002;
    </p>
    <h3 id="auto-23">2.5<span style="margin-left: 1em"></span>\(\langle L \nabla f, e_j \rangle\)&#x4E0E;
    \(\nabla H\)&#x7684;&#x5173;&#x7CFB;<span style="margin-left: 1em"></span></h3>
    <p>
      &#x8FD9;&#x662F;&#x524D;&#x9762;&#x8BA8;&#x8BBA;&#x7684;&#x6838;&#x5FC3;&#x6052;&#x7B49;&#x5F0F;&#x3002;&#x53D6;
      \(X = \nabla f\), \(H =\operatorname{Hess}f\).
    </p>
    <p>
      &#x7531;&#x8054;&#x7EDC; Laplacian&#x5B9A;&#x4E49;
    </p>
    <center>
      \(\displaystyle \langle L \nabla f, e_j \rangle = \sum_i \left[ \langle
      \nabla_{e_i}
\nabla_{e_i} \nabla f, e_j \rangle - \left\langle
      \nabla_{\nabla_{e_i} e_i}
\nabla f, e_j \right\rangle \right] .\)
    </center>
    <p>
      &#x5229;&#x7528;
    </p>
    <center>
      \(\displaystyle \langle \nabla_{e_i} \nabla f, e_j \rangle = H (e_i,
      e_j)\)
    </center>
    <p>
      &#x5E76;&#x5C55;&#x5F00;
    </p>
    <center>
      \(\displaystyle e_i (H (e_i, e_j)) = \langle \nabla_{e_i} \nabla_{e_i}
      \nabla f, e_j \rangle +
\langle \nabla_{e_i} \nabla f, \nabla_{e_i} e_j
      \rangle,\)
    </center>
    <p>
      &#x5F97;&#x5230;
    </p>
    <center>
      \(\displaystyle \langle \nabla_{e_i} \nabla_{e_i} \nabla f, e_j \rangle
      = e_i H (e_i, e_j) - H
(e_i, \nabla_{e_i} e_j) .\)
    </center>
    <p>
      &#x4EE3;&#x56DE;&#xFF1A;
    </p>
    <center>
      \(\displaystyle \langle L \nabla f, e_j \rangle = \sum_i [e_i H (e_i,
      e_j) - H (e_i,
\nabla_{e_i} e_j) - H (\nabla_{e_i} e_i, e_j)] .\)
    </center>
    <p>
      &#x65B9;&#x62EC;&#x53F7;&#x6B63;&#x662F; \((\nabla_{e_i} H) (e_i,
      e_j)\)&#xFF0C;&#x4E8E;&#x662F;
    </p>
    <center>
      \(\displaystyle \langle L \nabla f, e_j \rangle = \sum_i (\nabla_{e_i}
      H) (e_i, e_j) .\)
    </center>
    <p>
      &#x518D;&#x7531; \(H\) &#x5BF9;&#x79F0;&#xFF0C;
    </p>
    <center>
      \(\displaystyle \langle L \nabla f, e_j \rangle = \sum_i (\nabla_{e_i}
      H) (e_j, e_i) .\)
    </center>
    <h2 id="auto-24">3<span style="margin-left: 1em"></span>Bochner&#x516C;&#x5F0F;<span style="margin-left: 1em"></span></h2>
    <p>
      &#x4E3A;&#x4EC0;&#x4E48;&#x8981;&#x8BA1;&#x7B97;&#x68AF;&#x5EA6;&#x5E73;&#x65B9;&#x7684;Laplacian&#xFF1F;&#x56E0;&#x4E3A;&#x6211;&#x4EEC;&#x60F3;&#x628A;&ldquo;&#x51FD;&#x6570;&#x53D8;&#x5316;&#x5F97;&#x6709;&#x591A;&#x5FEB;&rdquo;&#x53D8;&#x6210;&#x4E00;&#x4E2A;&#x80FD;&#x7528;&#x6700;&#x5927;&#x503C;&#x539F;&#x7406;&#x63A7;&#x5236;&#x7684;&#x6807;&#x91CF;&#x95EE;&#x9898;&#x3002;Bochner&#x516C;&#x5F0F;&#x628A;&#x8FD9;&#x4E2A;&#x6807;&#x91CF;&#x7684;Laplacian&#x5206;&#x89E3;&#x6210;
      Hessian
      &#x7684;&#x975E;&#x8D1F;&#x5E73;&#x65B9;&#x9879;&#x3001;&#x539F;&#x65B9;&#x7A0B;&#x63D0;&#x4F9B;&#x7684;&#x9879;&#x4EE5;&#x53CA;&#x66F2;&#x7387;&#x9879;&#x3002;
    </p>
    <p>
      &#x4EE5;&#x4E0B;&#x8BBE; \((M^n, g)\)
      &#x662F;&#x5149;&#x6ED1;&#x9ECE;&#x66FC;&#x6D41;&#x5F62;&#xFF0C;\(f \in
      C^{\infty} (M)\)&#xFF0C;\(\nabla\) &#x662F; Levi-Civita
      &#x8054;&#x7EDC;&#x3002;&#x7EDF;&#x4E00;&#x91C7;&#x7528; \(\Delta
      =\operatorname{div}\operatorname{grad}\)&#xFF0C;&#x5E76;&#x53D6; \(R (X,
      Y) Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X, Y]}
      Z\)&#xFF0C;&#x4F7F;&#x5355;&#x4F4D;&#x7403;&#x9762;&#x7684;&#x622A;&#x9762;&#x66F2;&#x7387;&#x4E3A;&#x6B63;&#x3002;
    </p>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49;.
      </strong><i>Bochner&#x6052;&#x7B49;&#x5F0F;</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \frac{1}{2} \Delta | \nabla f |^2 = |
        \operatorname{Hess}f |^2 + \langle
\nabla f, \nabla \Delta f \rangle
        +\operatorname{Ric} (\nabla f, \nabla f)\)
      </center></i>
    </p>
    <p>
      &#x6B27;&#x6C0F;&#x7A7A;&#x95F4;&#x60C5;&#x5F62;
    </p>
    <p>
      &#x8BB0; \(f_j = \partial_j f\)&#xFF0C;\(f_{i j} = \partial_i \partial_j
      f\)&#x3002;&#x5BF9;&#x5E73;&#x65B9;&#x8FDE;&#x7EED;&#x6C42;&#x4E24;&#x6B21;&#x5BFC;&#x6570;
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  \frac{1}{2} \Delta | \nabla f |^2 &
      = \frac{1}{2} \sum_i \partial_i^2 \left(
  \sum_j f_j^2 \right)\\
  & =
      \sum_{i, j} f_{i j}^2 + \sum_j f_j \sum_i \partial_i^2 \partial_j f\\

      & = | D^2 f |^2 + \langle \nabla f, \nabla \Delta f \rangle
      .
\end{array}\)
    </center>
    <p>
      &#x5728;&#x6D41;&#x5F62;&#x4E0A;&#xFF0C;&#x5148;&#x5BF9;&#x5411;&#x91CF;&#x573A;&#x7684;&#x957F;&#x5EA6;&#x5E73;&#x65B9;&#x6C42;&#x5BFC;
    </p>
    <p>
      &#x4EE4;\(V = \nabla
      f\)&#x3002;&#x4E3A;&#x533A;&#x5206;&#x6807;&#x91CF;Laplacian&#x4E0E;&#x5411;&#x91CF;&#x573A;&#x4E0A;&#x7684;&#x7B97;&#x5B50;&#xFF0C;&#x8BB0;
    </p>
    <center>
      \(\displaystyle L V := \sum_i \left( \nabla_{e_i} \nabla_{e_i} V -
      \nabla_{\nabla_{e_i} e_i} V
\right) .\)
    </center>
    <p>
      &#x8FD9;&#x91CC; \(\{ e_i \}\)
      &#x662F;&#x5C40;&#x90E8;&#x6B63;&#x4EA4;&#x5355;&#x4F4D;&#x6807;&#x67B6;&#xFF0C;\(L\)
      &#x662F;&#x4E8C;&#x9636;&#x534F;&#x53D8;&#x5BFC;&#x6570;&#x7684;&#x8FF9;&#x3002;&#x56FA;&#x5B9A;&#x4E00;&#x70B9;\(p\)&#xFF0C;&#x9009;&#x53D6;&#x6EE1;&#x8DB3;
      \(\nabla_{e_i} e_j (p) = 0\)
      &#x7684;&#x6807;&#x67B6;&#x3002;&#x5229;&#x7528;&#x8054;&#x7EDC;&#x548C;&#x5EA6;&#x91CF;&#x76F8;&#x5BB9;&#xFF0C;&#x5728;
      \(p\) &#x6709;
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  \frac{1}{2} \Delta | V |^2 & =
      \sum_i e_i \langle \nabla_{e_i} V, V
  \rangle\\
  & = \sum_i |
      \nabla_{e_i} V |^2 + \left\langle \sum_i \nabla_{e_i}
  \nabla_{e_i} V,
      V \right\rangle\\
  & = | \nabla V |^2 + \langle L V, V
      \rangle
\end{array}\)
    </center>
    <p>
      &#x5F53; \(V = \nabla f\) &#x65F6;&#xFF0C;\(| \nabla V |^2 = |
      \operatorname{Hess}f
      |^2\)&#x3002;&#x56E0;&#x6B64;&#xFF0C;&#x5269;&#x4E0B;&#x7684;&#x95EE;&#x9898;&#x53EA;&#x6709;&#x4E00;&#x4E2A;&#xFF1A;\(L
      \nabla f\) &#x4E0E; \(\nabla \Delta f\)
      &#x76F8;&#x5DEE;&#x4EC0;&#x4E48;&#xFF1F;
    </p>
    <b><p>
      &#x4EA4;&#x6362;&#x5BFC;&#x6570;&#x4EA7;&#x751F;Ricci&#x66F2;&#x7387;
    </p></b>
    <center>
      \(\displaystyle L \nabla f = \nabla \Delta f +\operatorname{Ric} (\nabla
      f, \cdot)^{\sharp}\)
    </center>
    <p>
      \(\sharp\)
      &#x8868;&#x793A;&#x7528;&#x5EA6;&#x91CF;&#x628A;&#x4E00;&#x5F62;&#x5F0F;&#x53D8;&#x6210;&#x5411;&#x91CF;&#x3002;&#x5C06;&#x8FD9;&#x4E2A;&#x6052;&#x7B49;&#x5F0F;&#x4EE3;&#x5165;&#x4E0A;&#x9762;&#x7684;&#x5E73;&#x65B9;&#x6C42;&#x5BFC;&#x516C;&#x5F0F;&#xFF0C;&#x5C31;&#x5F97;&#x5230;Bochner&#x516C;&#x5F0F;&#x3002;&#x4E0B;&#x9762;&#x5C55;&#x5F00;&#x5B83;&#x7684;&#x8BC1;&#x660E;&#x3002;
    </p>
    <p style="margin-top: 1em">
      <strong>&#x8BC1;&#x660E;. </strong>
    </p>
    <p>
      &#x4EE4; \(H =\operatorname{Hess}f = \nabla d f\).
      &#x4ECD;&#x4F7F;&#x7528;&#x4E0A;&#x9762;&#x7684;&#x6807;&#x67B6;&#x3002;&#x7531;Hessian&#x7684;&#x5BF9;&#x79F0;&#x6027;
    </p>
    <center>
      \(\displaystyle \langle L \nabla f, e_j \rangle = \sum_i (\nabla_{e_i}
      H) (e_i, e_j) = \sum_i
(\nabla_{e_i} H) (e_j, e_i)\)
    </center>
    <p>
      &#x5BF9;&#x4E00;&#x5F62;&#x5F0F;
      \(\alpha\)&#xFF0C;&#x534F;&#x53D8;&#x5BFC;&#x6570;&#x7684;&#x53D8;&#x6362;&#x516C;&#x5F0F;&#x662F;
    </p>
    <center>
      \(\displaystyle ((\nabla_X \nabla_Y - \nabla_Y \nabla_X - \nabla_{[X,
      Y]}) \alpha) (Z) = -
\alpha (R (X, Y) Z) .\)
    </center>
    <p>
      &#x53D6; \(\alpha = d
      f\)&#xFF0C;&#x5E76;&#x5229;&#x7528;&#x8054;&#x7EDC;&#x65E0;&#x6320;&#xFF0C;&#x5F97;&#x5230;
    </p>
    <center>
      \(\displaystyle (\nabla_{e_i} H) (e_j, e_i) - (\nabla_{e_i} H) (e_i,
      e_i) = - \langle \nabla
f, R (e_i, e_j) e_i \rangle .\)
    </center>
    <p>
      &#x4E8E;&#x662F;
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  \langle L \nabla f, e_j \rangle & =
      \sum_i (\nabla_{e_j} H) (e_i, e_i) -
  \sum_i \langle \nabla f, R (e_i,
      e_j) e_i \rangle\\
  & = e_j (\Delta f) +\operatorname{Ric} (e_j, \nabla
      f) .
\end{array}\)
    </center>
    <p>
      &#x7B2C;&#x4E00;&#x9879;&#x662F; Hessian
      &#x7684;&#x8FF9;&#x7684;&#x5BFC;&#x6570;&#x3002;&#x7B2C;&#x4E8C;&#x9879;&#x7528;&#x5230;&#x66F2;&#x7387;&#x7B97;&#x5B50;&#x5173;&#x4E8E;&#x5EA6;&#x91CF;&#x7684;&#x53CD;&#x5BF9;&#x79F0;&#x6027;&#x4E0E;
      Ricci &#x66F2;&#x7387;&#x7684;&#x5B9A;&#x4E49;:
    </p>
    <center>
      \(\displaystyle - \sum_i \langle \nabla f, R (e_i, e_j) e_i \rangle =
      \sum_i \langle R (e_i,
e_j) \nabla f, e_i \rangle =\operatorname{Ric}
      (e_j, \nabla f) .\)
    </center>
    <p style="margin-bottom: 1em">
      &#x8FD9;&#x5BF9;&#x6BCF;&#x4E2A; \(e_j\)
      &#x90FD;&#x6210;&#x7ACB;&#xFF0C;&#x6240;&#x4EE5;&#x7ED9;&#x51FA;&#x4E86;&#x6240;&#x9700;&#x7684;&#x5411;&#x91CF;&#x6052;&#x7B49;&#x5F0F;&#x3002;<span
      style="margin-left: 1em"></span>\(\Box\)
    </p>
    <h3 id="auto-25">3.1<span style="margin-left: 1em"></span>&#x4E3A;&#x4EC0;&#x4E48;&#x9009;&#x62E9;&#x68AF;&#x5EA6;&#x5E73;&#x65B9;&#x7684;Laplacian<span
    style="margin-left: 1em"></span></h3>
    <p>
      &#x5047;&#x8BBE; \(u\)
      &#x6EE1;&#x8DB3;&#x4E00;&#x4E2A;&#x65B9;&#x7A0B;&#xFF0C;&#x5982;
      \(\Delta u =
      0\)&#xFF0C;&#x76EE;&#x6807;&#x662F;&#x4F30;&#x8BA1;&#x5B83;&#x7684;&#x53D8;&#x5316;&#x7387;
      \(| \nabla u
      |\)&#x3002;&#x8FD9;&#x4E2A;&#x9009;&#x62E9;&#x6709;&#x4E09;&#x4E2A;&#x5177;&#x4F53;&#x539F;&#x56E0;&#xFF1A;
    </p>
    <ul>
      <li>
        <p>
          &#x53D6;&#x68AF;&#x5EA6;&#xFF1A;&#x76F4;&#x63A5;&#x5BF9;&#x5E94;&#x6211;&#x4EEC;&#x60F3;&#x63A7;&#x5236;&#x7684;&#x53D8;&#x5316;&#x7387;
        </p>
      </li>
      <li>
        <p>
          &#x53D6;&#x5E73;&#x65B9;&#x8303;&#x6570;&#xFF1A;&#x628A;&#x5411;&#x91CF;&#x6362;&#x6210;&#x6807;&#x91CF;&#xFF0C;&#x4E14;
          \(| \nabla u
          |^2\)&#x5728;&#x4E34;&#x754C;&#x70B9;&#x5149;&#x6ED1;&#xFF1B;\(|
          \nabla u
          |\)&#x672A;&#x5FC5;&#x5982;&#x6B64;&#x3002;&#x5E73;&#x65B9;&#x6C42;&#x5BFC;&#x8FD8;&#x4EA7;&#x751F;&#x6709;&#x5229;&#x7684;Hessian&#x5E73;&#x65B9;&#x9879;&#x3002;
        </p>
      </li>
      <li>
        <p>
          &#x53D6;
          Laplacian&#xFF1A;&#x80FD;&#x8FDE;&#x63A5;&#x539F;&#x65B9;&#x7A0B;&#xFF0C;&#x5E76;&#x4E14;&#x5728;&#x6807;&#x91CF;&#x51FD;&#x6570;&#x7684;&#x5185;&#x90E8;&#x5C40;&#x90E8;&#x6700;&#x5927;&#x70B9;&#x6709;
          \(\Delta w \leq
          0\)&#xFF0C;&#x53EF;&#x4EE5;&#x4F7F;&#x7528;&#x6700;&#x5927;&#x503C;&#x539F;&#x7406;&#x3002;
        </p>
      </li>
    </ul>
    <p>
      &#x4F8B;&#x5982;&#xFF0C;&#x82E5; \(u\) &#x8C03;&#x548C;&#x4E14;
      \(\operatorname{Ric} \geq 0\)&#xFF0C;&#x5219;
    </p>
    <center>
      \(\displaystyle \frac{1}{2} \Delta | \nabla u |^2 = |
      \operatorname{Hess}u |^2
+\operatorname{Ric} (\nabla u, \nabla u) \geq
      0.\)
    </center>
    <p>
      &#x68AF;&#x5EA6;&#x5E73;&#x65B9;&#x56E0;&#x6B64;&#x662F;&#x6B21;&#x8C03;&#x548C;&#x51FD;&#x6570;&#xFF1A;&#x51E0;&#x4F55;&#x6761;&#x4EF6;&#x5DF2;&#x53D8;&#x6210;&#x6807;&#x91CF;&#x5FAE;&#x5206;&#x4E0D;&#x7B49;&#x5F0F;&#x3002;
    </p>
    <p>
      &#x4F46;&#x4EC5;&#x51ED;&#x8FD9;&#x4E00;&#x70B9;&#xFF0C;&#x4E0D;&#x80FD;&#x5728;&#x4EFB;&#x610F;&#x975E;&#x7D27;&#x6D41;&#x5F62;&#x4E0A;&#x76F4;&#x63A5;&#x5F97;&#x5230;&#x68AF;&#x5EA6;&#x4E0A;&#x754C;&#xFF0C;&#x56E0;&#x4E3A;&#x5B83;&#x672A;&#x5FC5;&#x53D6;&#x5F97;&#x6700;&#x5927;&#x503C;&#xFF0C;&#x4E5F;&#x7F3A;&#x5C11;&#x80FD;&#x9650;&#x5236;&#x5176;&#x5927;&#x5C0F;&#x7684;&#x9879;&#x3002;
    </p>
    <h3 id="auto-26">3.2<span style="margin-left: 1em"></span>&#x7528;&#x4E8E;&#x68AF;&#x5EA6;&#x4F30;&#x8BA1;&#xFF1A;&#x6B63;&#x7684;&#x4E8C;&#x6B21;&#x9879;&#x600E;&#x4E48;&#x51FA;&#x73B0;<span
    style="margin-left: 1em"></span></h3>
    <p>
      &#x8BBE;\(u\)&#x5728;&#x6240;&#x8BA8;&#x8BBA;&#x7684;&#x5F00;&#x96C6;&#x4E0A;&#x5149;&#x6ED1;&#x3001;&#x4E25;&#x683C;&#x4E3A;&#x6B63;&#x4E14;&#x8C03;&#x548C;&#xFF0C;&#x5373;
      \(u > 0, \Delta u = 0\)&#x3002;&#x4EE4; \(f = \log u, w = | \nabla f
      |^2\)&#x3002;&#x53D6;&#x5BF9;&#x6570;&#x662F;&#x4E3A;&#x4E86;&#x63A7;&#x5236;&#x76F8;&#x5BF9;&#x53D8;&#x5316;&#x7387;:
      \(u \mapsto c u (c >
      0)\)&#x53EA;&#x4F7F;\(f\)&#x589E;&#x52A0;&#x5E38;&#x6570;&#xFF0C;&#x4E0D;&#x6539;&#x53D8;&#x5B83;&#x7684;&#x68AF;&#x5EA6;&#x3002;&#x4E0B;&#x9762;&#x9010;&#x6B65;&#x8BA1;&#x7B97;
      \(\Delta f\)&#x3002;
    </p>
    <p>
      &#x53D6; \(a = u^{- 1}, V = \nabla u\)&#x3002;&#x6CE8;&#x610F; \(\nabla
      (u^{- 1}) = - u^{- 2} \nabla u\)&#x3002;&#x4E8E;&#x662F;
    </p>
    <center>
      \(\displaystyle \begin{array}{rl}
  \Delta f & =\operatorname{div}
      (\nabla f) =\operatorname{div} (u^{- 1}
  \nabla u)\\
  & = \langle
      \nabla (u^{- 1}), \nabla u \rangle + u^{- 1} \operatorname{div}

      (\nabla u)\\
  & = - u^{- 2} \langle \nabla u, \nabla u \rangle + u^{-
      1} \Delta u\\
  & = \frac{\Delta u}{u} - \frac{| \nabla u |^2}{u^2} = -
      \frac{| \nabla u
  |^2}{u^2}\\
  & = - w
\end{array}\)
    </center>
    <p>
      &#x56E0;&#x6B64;&#xFF0C;
    </p>
    <center>
      \(\displaystyle \Delta f = - | \nabla f |^2 = - w\)
    </center>
    <p>
      &#x6240;&#x4EE5; \(u\)
      &#x8C03;&#x548C;&#x5E76;&#x4E0D;&#x610F;&#x5473;&#x7740; \(\log u\)
      &#x8C03;&#x548C;&#xFF1B;&#x5BF9;&#x6570;&#x4EA7;&#x751F;&#x7684;&#x68AF;&#x5EA6;&#x5E73;&#x65B9;&#x9879;&#x6070;&#x597D;&#x628A;&#x7EBF;&#x6027;&#x65B9;&#x7A0B;
      \(\Delta u = 0\)
      &#x53D8;&#x6210;&#x540E;&#x7EED;&#x4F30;&#x8BA1;&#x6240;&#x9700;&#x7684;&#x975E;&#x7EBF;&#x6027;&#x65B9;&#x7A0B;&#x3002;&#x8D1F;&#x53F7;&#x6765;&#x81EA;
      \((\log s)'' = - s^{-
      2}\)&#xFF0C;&#x4E0D;&#x662F;&#x6765;&#x81EA;&#x66F2;&#x7387;&#x3002;
    </p>
    <p>
      &#x6700;&#x540E;&#xFF0C;&#x4E3A;&#x770B;&#x6E05;&#x5B83;&#x600E;&#x6837;&#x4EE3;&#x5165;
      Bochner &#x516C;&#x5F0F;&#xFF0C;&#x5BF9; \(\Delta f = - w\)
      &#x518D;&#x53D6;&#x68AF;&#x5EA6;&#xFF1A;
    </p>
    <center>
      \(\displaystyle \nabla \Delta f = - \nabla w, \quad \langle \nabla f,
      \nabla \Delta f \rangle
= - \langle \nabla f, \nabla w \rangle .\)
    </center>
    <p>
      &#x5BF9; \(f\) &#x4F7F;&#x7528; Bochner
      &#x516C;&#x5F0F;&#xFF0C;&#x4FBF;&#x6709;
    </p>
    <center>
      \(\displaystyle \frac{1}{2} \Delta w = | \operatorname{Hess}f |^2 -
      \langle \nabla f, \nabla w
\rangle +\operatorname{Ric} (\nabla f, \nabla
      f) .\)
    </center>
    <p>

    </p>
    <p>
      &#x5728;&#x4E00;&#x70B9;&#x628A;&#x5BF9;&#x79F0;&#x7684; Hessian
      &#x5BF9;&#x89D2;&#x5316;&#xFF0C;&#x8BBE;&#x5176;&#x7279;&#x5F81;&#x503C;&#x4E3A;
      \(\lambda_1, \ldots, \lambda_n\)&#x3002;Cauchy-Schwarz
      &#x4E0D;&#x7B49;&#x5F0F;&#x7ED9;&#x51FA;
    </p>
    <center>
      \(\displaystyle (\Delta f)^2 = \left( \sum_i \lambda_i \right)^2 \leq n
      \sum_i \lambda_i^2 = n
| \operatorname{Hess}f |^2 .\)
    </center>
    <p>
      &#x53C8;&#x56E0; \(\Delta f = - w\)&#xFF0C;&#x5F97;&#x5230; \(|
      \operatorname{Hess}f |^2 \geq w^2 / n\).
    </p>
    <p>
      &#x82E5; \(\operatorname{Ric} \geq - (n - 1) K
      g\)&#xFF0C;&#x5176;&#x4E2D; \(K \geq 0\)&#xFF0C;&#x5219;
    </p>
    <p style="margin-top: 1em">
      <strong>&#x5B9A;&#x4E49;.
      </strong><i>&#x68AF;&#x5EA6;&#x4F30;&#x8BA1;&#x7684;&#x6838;&#x5FC3;&#x5FAE;&#x5206;&#x4E0D;&#x7B49;&#x5F0F;</i>
    </p>
    <p style="margin-bottom: 1em">
      <i><center>
        \(\displaystyle \frac{1}{2} \Delta w \geq \frac{w^2}{n} - (n - 1) K w
        - \langle \nabla f,
\nabla w \rangle .\)
      </center></i>
    </p>
    <p>
      &#x6B63;&#x9879; \(w^2 / n\)
      &#x662F;&#x9650;&#x5236;&#x68AF;&#x5EA6;&#x4E0D;&#x80FD;&#x592A;&#x5927;&#x7684;&#x5173;&#x952E;&#x3002;&#x5047;&#x5982;
      \(w\)
      &#x5728;&#x5185;&#x90E8;&#x53D6;&#x5F97;&#x5C40;&#x90E8;&#x6700;&#x5927;&#x503C;&#xFF0C;&#x90A3;&#x4E48;&#x8BE5;&#x70B9;
      \(\nabla w = 0, \Delta w \leq 0\)&#xFF0C;&#x4ECE;&#x800C;
    </p>
    <center>
      \(\displaystyle 0 \geq \frac{w^2}{n} - (n - 1) K w, \quad w \leq n (n -
      1) K.\)
    </center>
    <p>
      &#x8FD9;&#x662F;&#x6700;&#x5927;&#x70B9;&#x5904;&#x7684;&#x6761;&#x4EF6;&#x6027;&#x7ED3;&#x8BBA;&#xFF0C;&#x4E0D;&#x662F;&#x5BF9;&#x4EFB;&#x610F;&#x5C40;&#x90E8;&#x8C03;&#x548C;&#x51FD;&#x6570;&#x5DF2;&#x7ECF;&#x5B8C;&#x6210;&#x7684;&#x4F30;&#x8BA1;&#x3002;&#x5B9E;&#x9645;&#x8BC1;&#x660E;&#x8981;&#x5728;&#x7403;&#x5185;&#x5BF9;&#x622A;&#x65AD;&#x540E;&#x7684;\(\psi
      w\)
      &#x4F7F;&#x7528;&#x6700;&#x5927;&#x503C;&#x539F;&#x7406;&#xFF0C;&#x63A7;&#x5236;&#x622A;&#x65AD;&#x4EA7;&#x751F;&#x7684;&#x8BEF;&#x5DEE;&#xFF0C;&#x6700;&#x7EC8;&#x5F97;&#x5230;
      \(| \nabla \log u | \leq C_n \left( \sqrt{K} + 1 / R
      \right)\)&#x7684;&#x5185;&#x90E8;&#x4F30;&#x8BA1;&#x3002;
    </p>
    <p>

    </p>
    <p>
      &#x7528;&#x4E8E;&#x8DDD;&#x79BB;&#x6BD4;&#x8F83;&#xFF1A;&#x68AF;&#x5EA6;&#x957F;&#x5EA6;&#x5DF2;&#x77E5;&#x65F6;&#x53CD;&#x8FC7;&#x6765;&#x7528;
    </p>
    <p>
      &#x8BBE;&#x6D41;&#x5F62;&#x5B8C;&#x5907;&#x3001;&#x8FDE;&#x901A;&#x3001;&#x65E0;&#x8FB9;&#x754C;&#x4E14;
      \(n \geq 2\)&#xFF0C;&#x53D6; \(r = d (p,
      \cdot)\)&#x3002;&#x53EA;&#x5728; \(M\backslash (\{ p \} \cup
      \operatorname{Cut} (p))\)
      &#x4E0A;&#x4F5C;&#x4E0B;&#x9762;&#x7684;&#x70B9;&#x6001;&#x8BA1;&#x7B97;&#x3002;&#x7531;&#x4E8E;
      \(| \nabla r |^2 =
      1\)&#xFF0C;Bochner&#x516C;&#x5F0F;&#x5DE6;&#x8FB9;&#x4E3A;&#x96F6;&#xFF1A;
    </p>
    <center>
      \(\displaystyle 0 = | \operatorname{Hess}r |^2 + \partial_r (\Delta r)
      +\operatorname{Ric}
(\partial_r, \partial_r), \quad \partial_r = \nabla
      r.\)
    </center>
    <p>
      &#x7531; \(\operatorname{Hess}r (\partial_r, \cdot) =
      0\)&#xFF0C;Hessian&#x7684;&#x8FF9;&#x53EA;&#x5728; \(n - 1\)
      &#x4E2A;&#x6A2A;&#x5411;&#x65B9;&#x5411;&#x4E0A;&#x6C42;&#x548C;&#xFF0C;&#x56E0;&#x6B64;
    </p>
    <center>
      \(\displaystyle | \operatorname{Hess}r |^2 \geq \frac{(\Delta r)^2}{n -
      1} .\)
    </center>
    <p>
      &#x82E5; \(\operatorname{Ric} \geq (n - 1) \kappa
      g\)&#xFF0C;&#x6CBF;&#x4ECE; \(p\)
      &#x51FA;&#x53D1;&#x3001;&#x521D;&#x59CB;&#x65B9;&#x5411;&#x662F; \(v\)
      &#x7684;&#x5F84;&#x5411;&#x6D4B;&#x5730;&#x7EBF; \(\gamma_v (t)\),
      &#x4EE4; \(m (t) = \Delta r (\gamma_v
      (t))\)&#x3002;&#x5728;&#x5230;&#x8FBE;&#x5272;&#x8FF9;&#x4E4B;&#x524D;
      (&#x5373; \(0 < t < c (v)\)&#xFF0C;&#x5176;&#x4E2D; \(c (v)\)
      &#x662F;&#x8BE5;&#x65B9;&#x5411;&#x7684;&#x5272;&#x65F6;&#x95F4;)&#xFF0C;&#x8DDD;&#x79BB;&#x51FD;&#x6570;
      \(r\) &#x5149;&#x6ED1;&#xFF0C;&#x56E0;&#x6B64;&#x5F97;&#x5230;
    </p>
    <center>
      \(\displaystyle m' + \frac{m^2}{n - 1} + (n - 1) \kappa \leq 0.\)
    </center>
    <p>
      &#x8FD9;&#x6B63;&#x662F;Laplacian&#x6BD4;&#x8F83;&#x6240;&#x9700;&#x7684;Riccati&#x578B;&#x4E0D;&#x7B49;&#x5F0F;&#x3002;&#x5BF9;&#x8DDD;&#x79BB;&#x51FD;&#x6570;&#xFF0C;&#x68AF;&#x5EA6;&#x957F;&#x5EA6;&#x5DF2;&#x77E5;&#xFF0C;&#x7528;
      Bochner &#x516C;&#x5F0F;&#x63A7;&#x5236; \(\Delta
      r\)&#xFF1B;&#x5BF9;&#x6B63;&#x8C03;&#x548C;&#x51FD;&#x6570;&#xFF0C;&#x65B9;&#x7A0B;&#x5DF2;&#x77E5;&#xFF0C;&#x7528;&#x5B83;&#x63A7;&#x5236;
      \(| \nabla \log u |\)&#x3002;&#x5B8C;&#x6574;&#x77E9;&#x9635; Riccati
      &#x65B9;&#x7A0B;&#x8FD8;&#x9700; Jacobi
      &#x573A;&#xFF0C;&#x4E0D;&#x80FD;&#x4EC5;&#x7531;&#x8FD9;&#x6761;&#x53D6;&#x8FF9;&#x7684;&#x6052;&#x7B49;&#x5F0F;&#x6062;&#x590D;&#x3002;
    </p>
    <p>

    </p>
  </body>

<!-- prettier-ignore-end -->
