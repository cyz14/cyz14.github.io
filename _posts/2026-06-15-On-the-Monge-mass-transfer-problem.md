---
layout: post
title: On the Monge mass transfer problem, Trudinger & Xu-Jia Wang
date: 2026-06-15 17:00:00
description: note of this paper
tags: ot
categories: math
---

<body>
    <table class="title-block" style="margin-bottom: 2em">
      <tr>
        <td><table class="title-block" style="margin-top: 0.5em; margin-bottom: 0.5em;">
          <tr>
            <td style="text-align: center;"><strong>On the Monge mass transfer problem</strong></td>
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
                        <class style="font-variant: small-caps">by Neil S. Trudinger, Xu-Jia
                        Wang</class>
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
      <a id="auto-1"></a>
    </p>
    <center>
      <p style="margin-top: 0.5em; margin-bottom: 0.5em">
        <strong>Abstract</strong>
      </p>
    </center>
    <p>
      蒙日在1781年提出的质量传输问题，是从一个质量分布移动到另一个使得一个代价泛函在所有保测度映射中最小。最优传输映射的存在性在1979年由Sudakov用概率论证明。最近Evans和Gangbo发现了一个基于PDE的证明。这篇论文中作者们通过直接从蒙日和康塔洛维奇的势能函数构造一个最优传输的方式给出一个更基本和更短的证明。
    </p>
    <h2 id="auto-2">1<span style="margin-left: 1em"></span>Introduction<span style="margin-left: 1em"></span></h2>
    <p>
      给定欧氏空间 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>ℝ</mi><mi>n</mi></msup></math>
      上的两个有界开集 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mo>,</mo><mi>V</mi></mrow></math>，和对应的有相同总质量的质量分布，用非负函数<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>,</mo><mi>g</mi><mo>∈</mo><msup><mi>L</mi><mn>1</mn></msup><mrow><mo
      form="prefix">(</mo><mi>U</mi><mo form="postfix">)</mo></mrow><mo>,</mo><msup><mi>L</mi><mn>1</mn></msup><mrow><mo
      form="prefix">(</mo><mi>V</mi><mo form="postfix">)</mo></mrow></mrow></math>分别表示，满足
    </p>
    <table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><munder><mo>&Integral;</mo><mi>U</mi></munder><mi>f</mi><mo>=</mo><munder><mo>&Integral;</mo><mi>V</mi></munder><mi>g</mi></mrow></mstyle></math></td>
        <td align="right">(1.1)</td>
      </tr>
    </table>
    <p>
      蒙日问题是是否存在一个保测度映射<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mn>0</mn></msub><mo>:</mo><mi>U</mi><mo>→</mo><mi>V</mi></mrow></math>最小化这个代价泛函
    </p>
    <table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>𝒞</mi><mrow><mo
        form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>=</mo><munder><mo>&Integral;</mo><mi>U</mi></munder><mi>f</mi><mrow><mo
        form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mrow><mo form="prefix">|</mo><mi>s</mi><mrow><mo
        form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>-</mo><mi>x</mi><mo
        form="postfix">|</mo></mrow><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi></mrow></mstyle></math></td>
        <td align="right">(1.2)</td>
      </tr>
    </table>
    <p>
      在所有从<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>U</mi></math>到<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>V</mi></math>的保测度映射中。一个映射<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>s</mi></math>被叫做保测度的如果它是Borel可测的且对任意Borel
      集 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>E</mi><mo>⊂</mo><mi>V</mi></mrow></math>, 
    </p>
    <table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><munder><mo>&Integral;</mo><mrow><msup><mi>s</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mrow><mo
        form="prefix">(</mo><mi>E</mi><mo form="postfix">)</mo></mrow></mrow></munder><mi>f</mi><mo>=</mo><munder><mo>&Integral;</mo><mi>E</mi></munder><mi>g</mi><mo>,</mo></mrow></mstyle></math></td>
        <td align="right">(1.3)</td>
      </tr>
    </table>
    <p>
      即它相对于测度<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>&ApplyFunction;</mo><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi><mo>,</mo><mi>g</mi><mo>&ApplyFunction;</mo><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi></mrow></math>是保测度的。在早期蒙日自己的，Apell和Kantorovich的贡献之后，Sudakov在1979年确实地解决了，基于他对概率空间进行分解的研究成果。最近，Evans和Gangbo发现一个用PDE新的证明，在<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>f</mi></math>和<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>g</mi></math>的更强的假设，但和Sudakov一开始的证明一样，它非常复杂。这篇note的目标是提供一个更基本简短的证明。
    </p>
    <p>
      蒙日问题也已经对其他代价函数考虑过。给定任意连续代价函数<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>:</mo><msup><mi>ℝ</mi><mi>n</mi></msup><mo>×</mo><msup><mi>ℝ</mi><mi>n</mi></msup><mo>→</mo><msup><mi>ℝ</mi><mrow></mrow></msup></mrow></math>，我们定义一个对应的代价泛函<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>𝒞</mi></math>
    </p>
    <table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>𝒞</mi><mrow><mo
        form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>=</mo><munder><mo>&Integral;</mo><mi>U</mi></munder><mi>f</mi><mrow><mo
        form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>s</mi><mrow><mo
        form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi><mo>,</mo></mrow></mstyle></math></td>
        <td align="right">(1.4)</td>
      </tr>
    </table>
    <p>
      其中<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>𝒮</mi><mo>=</mo><mi>𝒮</mi><mrow><mo
      form="prefix">(</mo><mi>f</mi><mo>,</mo><mi>g</mi><mo form="postfix">)</mo></mrow></mrow></math>，所有从<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><mi>U</mi><mo>,</mo><mi>f</mi><mo>&ApplyFunction;</mo><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi><mo
      form="postfix">)</mo></mrow></math>到<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><mi>V</mi><mo>,</mo><mi>g</mi><mo>&ApplyFunction;</mo><mi>d</mi><mo>&ApplyFunction;</mo><mi>y</mi><mo
      form="postfix">)</mo></mrow></math>的保测度映射的集合。一个重要情形是
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo
      form="postfix">)</mo></mrow><mo>=</mo><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
      form="postfix">)</mo></mrow></mrow></math> 且<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>c</mi></math>是一个严格凸函数，所以
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>D</mi><mo>&ApplyFunction;</mo><mi>c</mi></mrow></math>
      可逆。在特殊的二次代价情形 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mo>=</mo><msup><mrow><mo
      form="prefix">|</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo form="postfix">|</mo></mrow><mn>2</mn></msup></mrow></math>，最优映射的正则性也已经证明了。
    </p>
    <p>
      我们对蒙日传输问题的方法利用了严格凸代价函数的近似，像Caffarelli，Gangbo和McCann那样，传输射线和传输集合的记号来自Evans和Gangbo。这些工作中，方法的关键依赖在于Kantorovich对偶问题的势能函数。我们的想法是构造一个映射
      <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mn>0</mn></msub></math>，满足质量平衡条件
      (1.3)，使得对于传输射线 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ℓ</mi></math>
      中的每个点 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mn>0</mn></msub><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow></mrow></math>也落在
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ℓ</mi></math> 上。而且，如果 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub><mo>∈</mo><mi>ℓ</mi></mrow></math>,
    </p>
    <table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mrow><mo
        form="prefix">(</mo><msub><mi>s</mi><mn>0</mn></msub><mrow><mo form="prefix">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo
        form="postfix">)</mo></mrow><mo>-</mo><msub><mi>s</mi><mn>0</mn></msub><mrow><mo
        form="prefix">(</mo><msub><mi>x</mi><mn>2</mn></msub><mo form="postfix">)</mo></mrow><mo
        form="postfix">)</mo></mrow><mo>⋅</mo><mrow><mo form="prefix">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo>-</mo><msub><mi>x</mi><mn>2</mn></msub><mo
        form="postfix">)</mo></mrow><mo>≥</mo><mn>0</mn><mi>.</mi></mrow></mstyle></math></td>
        <td align="right">(1.5)</td>
      </tr>
    </table>
    <p>
      我们证明这个映射 <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mn>0</mn></msub></math>
      确实是最优的。为了构造满足单调性条件(1.5)的映射
      <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mn>0</mn></msub></math>，我们会使用更简单的严格凸代价函数情形下
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mi>p</mi></msub><mrow><mo form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo
      form="postfix">)</mo></mrow><mo>=</mo><msup><mrow><mo form="prefix">|</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
      form="postfix">|</mo></mrow><mi>p</mi></msup><mo>,</mo><mi>p</mi><mo>></mo><mn>1</mn></mrow></math>的最优传输的存在性。更多蒙日问题的讨论，它有趣的历史和各种应用，读者可以参考[5,7,8,10,14,17]。
    </p>
    <h2 id="auto-3">2<span style="margin-left: 1em"></span>严格凸的情形<span style="margin-left: 1em"></span></h2>
    <p>
      要证明质量传输问题在(1.2)的代价函数下的最优传输的存在性，我们需要对应问题在严格凸代价函数下的存在性。我们保持<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>,</mo><mi>g</mi></mrow></math>非负Borel可测函数的假设，在
      <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>ℝ</mi><mi>n</mi></msup></math>
      上，支撑有界，满足质量平衡条件
    </p>
    <table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mo>&Integral;</mo><mi>f</mi><mo>=</mo><mo>&Integral;</mo><mi>g</mi><mo>&lt;</mo><mi>∞</mi></mrow></mstyle></math></td>
        <td align="right">(2.1)</td>
      </tr>
    </table>
    <p>
      令<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>𝒮</mi><mo>=</mo><mi>𝒮</mi><mrow><mo
      form="prefix">(</mo><mi>f</mi><mo>,</mo><mi>g</mi><mo form="postfix">)</mo></mrow></mrow></math>表示
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><msup><mi>ℝ</mi><mi>n</mi></msup><mo>,</mo><mi>μ</mi><mo
      form="postfix">)</mo></mrow></math> 到 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><msup><mi>ℝ</mi><mi>n</mi></msup><mo>,</mo><mi>ν</mi><mo
      form="postfix">)</mo></mrow></math> 的保测度映射集合，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>=</mo><mi>f</mi><mo>&ApplyFunction;</mo><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi><mo>,</mo><mi>ν</mi><mo>=</mo><mi>g</mi><mo>&ApplyFunction;</mo><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi></mrow></math>.
    </p>
    <p style="margin-top: 1em">
      <strong>Theorem <class style="font-style: normal">2.1</class>. </strong><i>Let <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>∈</mo><msup><mi>C</mi><mn>1</mn></msup><mrow><mo
      form="prefix">(</mo><msup><mi>ℝ</mi><mi>n</mi></msup><mo form="postfix">)</mo></mrow></mrow></math>
      be strictly convex. Then there exists an optimal mapping <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>s</mi><mi>c</mi></msub><mo>∈</mo><mi>𝒮</mi></mrow></math>
      such that</i>
    </p>
    <i><table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>𝒞</mi><mrow><mo
        form="prefix">(</mo><msub><mi>s</mi><mi>c</mi></msub><mo form="postfix">)</mo></mrow><mo>=</mo><msub><mo>inf</mo><mrow><mi>s</mi><mo>∈</mo><mi>𝒮</mi></mrow></msub><mi>𝒞</mi><mrow><mo
        form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>,</mo></mrow></mstyle></math></td>
        <td align="right">(2.2)</td>
      </tr>
    </table></i>
    <p>
      <i>where </i>
    </p>
    <i><table width="100%">
      <tr>
        <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>𝒞</mi><mrow><mo
        form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mo>&Integral;</mo><mi>f</mi><mrow><mo
        form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>s</mi><mrow><mo
        form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi><mi>.</mi></mrow></mstyle></math></td>
        <td align="right">(2.3)</td>
      </tr>
    </table></i>
    <p>
      <i>Furthermore the mapping <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mi>c</mi></msub></math>
      is invertible (almost everywhere), with inverse <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>s</mi><mi>c</mi><mrow><mo>-</mo><mn>1</mn></mrow></msubsup><mo>∈</mo><mi>𝒮</mi><mrow><mo
      form="prefix">(</mo><mi>g</mi><mo>,</mo><mi>f</mi><mo form="postfix">)</mo></mrow></mrow></math>
      a minimizer among <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>𝒮</mi><mrow><mo form="prefix">(</mo><mi>g</mi><mo>,</mo><mi>f</mi><mo
      form="postfix">)</mo></mrow></mrow></math> of the functional</i>
    </p>
    <div style="margin-bottom: 1em">
      <i><table width="100%">
        <tr>
          <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><msup><mi>𝒞</mi><mi>'</mi></msup><mrow><mo
          form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mo>&Integral;</mo><mi>g</mi><mrow><mo
          form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>c</mi><mrow><mo
          form="prefix">(</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>-</mo><mi>x</mi><mo
          form="postfix">)</mo></mrow><mi>.</mi></mrow></mstyle></math></td>
          <td align="right">(2.4)</td>
        </tr>
      </table></i>
    </div>
    <p style="margin-top: 1em">
      <strong>Proof.
      </strong>这里简单介绍证明思路，因为后面对<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mrow><mo form="prefix">|</mo><mi>x</mi><mo
      form="postfix">|</mo></mrow></mrow></math>的证明里还会用到；更多细节见[4,9,17].证明分4步。
    </p>
    <ol>
      <li>
        <p>
          考虑Kantorovich的对偶问题，最大化这个泛函
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>I</mi><mrow><mo
            form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mo>&Integral;</mo><mi>u</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>f</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>d</mi><mo>&ApplyFunction;</mo><mi>x</mi><mo>+</mo><mo>&Integral;</mo><mi>v</mi><mrow><mo
            form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mi>g</mi><mrow><mo
            form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mi>d</mi><mo>&ApplyFunction;</mo><mi>y</mi></mrow></mstyle></math></td>
            <td align="right">(2.5)</td>
          </tr>
        </table>
        <p>
          在所有下面集合的函数对中
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>𝒦</mi><mo>=</mo><mrow><mo
            form="prefix">{</mo><mrow><mo form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo
            form="postfix">)</mo></mrow><mo>∈</mo><mi>C</mi><mrow><mo form="prefix">(</mo><msup><mi>ℝ</mi><mi>n</mi></msup><mo
            form="postfix">)</mo></mrow><mo>|</mo><mi>u</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
            form="postfix">)</mo></mrow><mo>+</mo><mi>v</mi><mrow><mo form="prefix">(</mo><mi>y</mi><mo
            form="postfix">)</mo></mrow><mo>≤</mo><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
            form="postfix">)</mo></mrow><mo>&ApplyFunction;</mo><mo>∀</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>∈</mo><mi>Ω</mi><mo
            form="postfix">}</mo></mrow><mo>,</mo></mrow></mstyle></math></td>
            <td align="right">(2.6)</td>
          </tr>
        </table>
        <p>
          其中<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Ω</mi></math> 是一个 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>supp</mi><mo>&ApplyFunction;</mo><mi>f</mi><mo>∪</mo><mi>supp</mi><mo>&ApplyFunction;</mo><mi>g</mi></mrow></math>
          的有界开邻域。首先我们证明 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mrow><mo
          form="prefix">(</mo><mo>⋅</mo><mo>,</mo><mo>⋅</mo><mo form="postfix">)</mo></mrow></mrow></math>
          在 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>𝒦</mi></math> 中有一个最大化函数对
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><mi>φ</mi><mo>,</mo><mi>ψ</mi><mo
          form="postfix">)</mo></mrow></math>. 实际上，给定任意 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo
          form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></math>，令
        </p>
        <center>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mtable columnalign="right left">
            <mtr>
              <mtd><mrow><msup><mi>u</mi><mo>∗</mo></msup><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow></mrow></mtd>
              <mtd><mrow><mo>=</mo><msub><mo>inf</mo><mrow><mi>y</mi><mo>∈</mo><mi>Ω</mi></mrow></msub><mrow><mo
              form="prefix">{</mo><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow><mo>-</mo><mi>v</mi><mrow><mo form="prefix">(</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow><mo form="postfix">}</mo></mrow><mo>,</mo></mrow></mtd>
            </mtr>
            <mtr>
              <mtd><mrow><msup><mi>v</mi><mo>∗</mo></msup><mrow><mo form="prefix">(</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow></mrow></mtd>
              <mtd><mrow><mo>=</mo><msub><mo>inf</mo><mrow><mi>x</mi><mo>∈</mo><mi>Ω</mi></mrow></msub><mrow><mo
              form="prefix">{</mo><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow><mo>-</mo><msup><mi>u</mi><mo>∗</mo></msup><mrow><mo
              form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo form="postfix">}</mo></mrow><mi>.</mi></mrow></mtd>
            </mtr>
          </mtable></mstyle></math>
        </center>
        <p>
          那么 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>u</mi><mo>∗</mo></msup><mo>≥</mo><mi>u</mi><mo>,</mo><msup><mi>v</mi><mo>∗</mo></msup><mo>≥</mo><mi>v</mi><mo>,</mo><mrow><mo
          form="prefix">(</mo><msup><mi>u</mi><mo>∗</mo></msup><mo>,</mo><msup><mi>v</mi><mo>∗</mo></msup><mo
          form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></math>, 且
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mrow><mo form="prefix">(</mo><msup><mi>u</mi><mo>∗</mo></msup><mo>,</mo><msup><mi>v</mi><mo>∗</mo></msup><mo
          form="postfix">)</mo></mrow><mo>≥</mo><mi>I</mi><mrow><mo form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo
          form="postfix">)</mo></mrow></mrow></math>.而且，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>u</mi><mo>∗</mo></msup><mo>,</mo><msup><mi>v</mi><mo>∗</mo></msup></mrow></math>是
          Lipschitz连续的，他们的 Lipschitz 常数小于或等于 <math
          xmlns="http://www.w3.org/1998/Math/MathML"><mi>c</mi></math> 的。因此对任意序列 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo
          form="prefix">{</mo><mrow><mo form="prefix">(</mo><msub><mi>u</mi><mi>k</mi></msub><mo>,</mo><msub><mi>v</mi><mi>k</mi></msub><mo
          form="postfix">)</mo></mrow><mo form="postfix">}</mo></mrow><mo>⊂</mo><mi>𝒦</mi></mrow></math>
          使得 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mrow><mo form="prefix">(</mo><msub><mi>u</mi><mi>k</mi></msub><mo>,</mo><msub><mi>v</mi><mi>k</mi></msub><mo
          form="postfix">)</mo></mrow><mo>→</mo><msub><mo>sup</mo><mrow><mrow><mo
          form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></msub><mi>I</mi><mrow><mo
          form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow></mrow></math>，我们可以用
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><msubsup><mi>u</mi><mi>k</mi><mo>∗</mo></msubsup><mo>,</mo><msubsup><mi>v</mi><mi>k</mi><mo>∗</mo></msubsup><mo
          form="postfix">)</mo></mrow></math> 替换 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><msub><mi>u</mi><mi>k</mi></msub><mo>,</mo><msub><mi>v</mi><mi>k</mi></msub><mo
          form="postfix">)</mo></mrow></math>。根据
          (2.1)的质量守恒条件，我们有对任意常数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>C</mi></math>，<math
          xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo form="prefix">(</mo><mi>u</mi><mo>+</mo><mi>C</mi><mo>,</mo><mi>v</mi><mo>-</mo><mi>C</mi><mo
          form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></math>
          如果<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo
          form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></math>。因此，我们可以假设
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><msubsup><mi>u</mi><mi>k</mi><mo>∗</mo></msubsup><mo>,</mo><msubsup><mi>v</mi><mi>k</mi><mo>∗</mo></msubsup><mo
          form="postfix">)</mo></mrow></math>是一致有界的，因此，通过选择一个必要的子列，它收敛于一个函数对
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo form="prefix">(</mo><mi>φ</mi><mo>,</mo><mi>ψ</mi><mo
          form="postfix">)</mo></mrow><mo>⊂</mo><mi>𝒦</mi></mrow></math>.
          而且，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>φ</mi><mo>,</mo><mi>ψ</mi></mrow></math>
          是Lipschitz 连续的，它们的 Lipschitz 常数不大于 <math
          xmlns="http://www.w3.org/1998/Math/MathML"><mi>c</mi></math>的。
        </p>
      </li>
      <li>
        <p>
          从这个构造过程我们有
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>φ</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><msub><mo>inf</mo><mrow><mi>y</mi><mo>∈</mo><mi>Ω</mi></mrow></msub><mrow><mo
            form="prefix">{</mo><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
            form="postfix">)</mo></mrow><mo>-</mo><mi>ψ</mi><mrow><mo form="prefix">(</mo><mi>y</mi><mo
            form="postfix">)</mo></mrow><mo form="postfix">}</mo></mrow></mrow></mstyle></math></td>
            <td align="right">(2.7)</td>
          </tr>
        </table>
        <p>
          即，对任意 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>Ω</mi></mrow></math>，存在一个点
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
          form="postfix">)</mo></mrow><mo>∈</mo><mover><mi>Ω</mi><mo>&OverBar;</mo></mover></mrow></math>
          使得
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>φ</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mi>c</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
            form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mo>-</mo><mi>ψ</mi><mrow><mo
            form="prefix">(</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo
            form="postfix">)</mo></mrow></mrow></mstyle></math></td>
            <td align="right">(2.8)</td>
          </tr>
        </table>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>φ</mi><mrow><mo
            form="prefix">(</mo><msup><mi>x</mi><mi>'</mi></msup><mo form="postfix">)</mo></mrow><mo>≤</mo><mi>c</mi><mrow><mo
            form="prefix">(</mo><msup><mi>x</mi><mi>'</mi></msup><mo>-</mo><mi>s</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mo>-</mo><mi>ψ</mi><mrow><mo
            form="prefix">(</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo
            form="postfix">)</mo></mrow><mo>∀</mo><msup><mi>x</mi><mi>'</mi></msup><mo>∈</mo><mi>Ω</mi><mi>.</mi></mrow></mstyle></math></td>
            <td align="right">(2.9)</td>
          </tr>
        </table>
        <p>
          可得
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>D</mi><mo>&ApplyFunction;</mo><mi>φ</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mi>D</mi><mo>&ApplyFunction;</mo><mi>c</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
            form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mi>.</mi></mrow></mstyle></math></td>
            <td align="right">(2.10)</td>
          </tr>
        </table>
        <p>
          注意 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>φ</mi></math> 是 Lipschitz
          连续的所以几乎处处可微。因此 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>c</mi></math>
          严格凸，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>D</mi><mo>&ApplyFunction;</mo><mi>c</mi></mrow></math>
          可逆。因此
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><msub><mi>s</mi><mi>c</mi></msub><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>?</mo><mi>s</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mi>x</mi><mo>-</mo><msup><mrow><mo
            form="prefix">(</mo><mi>D</mi><mo>&ApplyFunction;</mo><mi>c</mi><mo form="postfix">)</mo></mrow><mrow><mo>-</mo><mn>1</mn></mrow></msup><mrow><mo
            form="prefix">(</mo><mi>D</mi><mo>&ApplyFunction;</mo><mi>φ</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mo>,</mo></mrow></mstyle></math></td>
            <td align="right">(2.11)</td>
          </tr>
        </table>
        <p>
          可以唯一确定且是一个一一映射，除了一个零测度集。
        </p>
      </li>
      <li>
        <p>
          下面我们证明映射 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>=</mo><msub><mi>s</mi><mi>c</mi></msub></mrow></math>
          (2.11) 满足保测度条件
          (1.3)。实际上，对任意连续函数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>h</mi></math>
          和小常数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ε</mi></math>，令
        </p>
        <center>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mtable columnalign="right left">
            <mtr>
              <mtd><mrow><msub><mi>ψ</mi><mi>ε</mi></msub><mrow><mo form="prefix">(</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow></mrow></mtd>
              <mtd><mrow><mo>=</mo><mi>ψ</mi><mrow><mo form="prefix">(</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow><mo>+</mo><mi>ε</mi><mi>h</mi><mrow><mo
              form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mo>,</mo></mrow></mtd>
            </mtr>
            <mtr>
              <mtd><mrow><msub><mi>φ</mi><mi>ε</mi></msub><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow></mrow></mtd>
              <mtd><mrow><mo>=</mo><msub><mo>inf</mo><mrow><mi>y</mi><mo>∈</mo><mi>Ω</mi></mrow></msub><mrow><mo
              form="prefix">{</mo><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo
              form="postfix">)</mo></mrow><mo>-</mo><msub><mi>ψ</mi><mi>ε</mi></msub><mrow><mo
              form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mo form="postfix">}</mo></mrow><mi>.</mi></mrow></mtd>
            </mtr>
          </mtable></mstyle></math>
        </center>
        <p>
          我们可以容易检验
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><msub><mi>φ</mi><mi>ε</mi></msub><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mi>φ</mi><mrow><mo
            form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>-</mo><mi>ε</mi><mi>h</mi><mrow><mo
            form="prefix">(</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo
            form="postfix">)</mo></mrow><mo>+</mo><mi>o</mi><mrow><mo form="prefix">(</mo><mi>ε</mi><mo
            form="postfix">)</mo></mrow><mo>&ApplyFunction;</mo><mi>as</mi><mo>&ApplyFunction;</mo><mi>ε</mi><mo>→</mo><mn>0</mn><mi>.</mi></mrow></mstyle></math></td>
            <td align="right">(2.12)</td>
          </tr>
        </table>
        <p>
          因为 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><mi>φ</mi><mo>,</mo><mi>ψ</mi><mo
          form="postfix">)</mo></mrow></math> 最大化泛函 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mrow><mo
          form="prefix">(</mo><mo>⋅</mo><mo>,</mo><mo>⋅</mo><mo form="postfix">)</mo></mrow></mrow></math>，我们可知
          
        </p>
        <center>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mtable columnalign="right left">
            <mtr>
              <mtd><mn>0</mn></mtd>
              <mtd><mrow><mo>=</mo><msub><mo>lim</mo><mrow><mi>ε</mi><mo>→</mo><mn>0</mn></mrow></msub><mfrac><mn>1</mn><mi>ε</mi></mfrac><mrow><mo
              form="prefix">(</mo><mi>I</mi><mrow><mo form="prefix">(</mo><msub><mi>φ</mi><mi>ε</mi></msub><mo>,</mo><msub><mi>ψ</mi><mi>ε</mi></msub><mo
              form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mo>-</mo><mi>I</mi><mrow><mo
              form="prefix">(</mo><mi>φ</mi><mo>,</mo><mi>ψ</mi><mo form="postfix">)</mo></mrow></mrow></mtd>
            </mtr>
            <mtr>
              <mtd><mrow></mrow></mtd>
              <mtd><mrow><mo>=</mo><mo>-</mo><mo>&Integral;</mo><mi>h</mi><mrow><mo
              form="prefix">(</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo
              form="postfix">)</mo></mrow><mi>f</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow><mo>+</mo><mo>&Integral;</mo><mi>h</mi><mrow><mo
              form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mi>g</mi><mrow><mo
              form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mi>d</mi><mo>&ApplyFunction;</mo><mi>y</mi><mo>,</mo><mtext>(2.13)</mtext></mrow></mtd>
            </mtr>
          </mtable></mstyle></math>
        </center>
        <p>
          这等价于 (1.3)。
        </p>
      </li>
      <li>
        <p>
          我们有，对任意 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>𝒮</mi><mo>,</mo><mrow><mo
          form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></math>，
        </p>
        <center>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mtable columnalign="right left">
            <mtr>
              <mtd><mrow><mo>&Integral;</mo><mi>u</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow><mi>f</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow><mo>+</mo><mo>&Integral;</mo><mi>v</mi><mrow><mo
              form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mi>g</mi><mrow><mo
              form="prefix">(</mo><mi>y</mi><mo form="postfix">)</mo></mrow></mrow></mtd>
              <mtd><mrow><mo>=</mo><mo>&Integral;</mo><mrow><mo form="prefix">(</mo><mi>u</mi><mrow><mo
              form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>f</mi><mrow><mo
              form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>+</mo><mi>v</mi><mrow><mo
              form="prefix">(</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo
              form="postfix">)</mo></mrow><mi>f</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow></mrow></mtd>
            </mtr>
            <mtr>
              <mtd><mrow></mrow></mtd>
              <mtd><mrow><mo>≤</mo><mo>&Integral;</mo><mi>c</mi><mrow><mo
              form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>s</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
              form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mi>f</mi><mrow><mo
              form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mi>.</mi><mtext>(2.14)</mtext></mrow></mtd>
            </mtr>
          </mtable></mstyle></math>
        </center>
        <p>
          即 
        </p>
        <table width="100%">
          <tr>
            <td align="center" width="100%"><math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><msub><mo>sup</mo><mrow><mrow><mo
            form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow><mo>∈</mo><mi>𝒦</mi></mrow></msub><mi>I</mi><mrow><mo
            form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo form="postfix">)</mo></mrow><mo>≤</mo><msub><mo>inf</mo><mrow><mi>s</mi><mo>∈</mo><mi>𝒮</mi></mrow></msub><mi>𝒞</mi><mrow><mo
            form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>,</mo></mrow></mstyle></math></td>
            <td align="right">(2.15)</td>
          </tr>
        </table>
        <p>
          等号成立当 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>=</mo><msub><mi>s</mi><mi>c</mi></msub></mrow></math>
          且 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo
          form="postfix">)</mo></mrow><mo>=</mo><mrow><mo form="prefix">(</mo><mi>φ</mi><mo>,</mo><mi>ψ</mi><mo
          form="postfix">)</mo></mrow></mrow></math>。因此，<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mi>c</mi></msub></math>
          是一个最优传输。从第2步我们知道 <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mi>c</mi></msub></math>
          可逆（几乎处处）。上面的证明思路，对 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>f</mi></math>
          和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>g</mi></math> 替换后，证明 <math xmlns="http://www.w3.org/1998/Math/MathML"><msubsup><mi>s</mi><mi>c</mi><mrow><mo>-</mo><mn>1</mn></mrow></msubsup></math>
          是 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>𝒮</mi><mrow><mo form="prefix">(</mo><mi>g</mi><mo>,</mo><mi>f</mi><mo
          form="postfix">)</mo></mrow></mrow></math> 中映射的泛函 (2.4)
          的最小者。
        </p>
      </li>
    </ol>
    <p style="margin-bottom: 1em">
      <span style="margin-left: 1em"></span><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>□</mi></math>
    </p>
    <p>
      定理2.1 的最优传输 <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>s</mi><mi>c</mi></msub></math>
      是唯一的，如果 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>∈</mo><mi>𝒮</mi><mrow><mo
      form="prefix">(</mo><mi>f</mi><mo>,</mo><mi>g</mi><mo form="postfix">)</mo></mrow></mrow></math>
      使得 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>𝒞</mi><mrow><mo form="prefix">(</mo><mi>t</mi><mo
      form="postfix">)</mo></mrow><mo>=</mo><mi>𝒞</mi><mrow><mo form="prefix">(</mo><msub><mi>s</mi><mi>c</mi></msub><mo
      form="postfix">)</mo></mrow></mrow></math>,则 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><msub><mi>s</mi><mi>c</mi></msub></mrow></math>
      几乎处处 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>supp</mi><mo>&ApplyFunction;</mo><mi>f</mi></mrow></math>。实际上，令(2.14)中
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo form="prefix">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo
      form="postfix">)</mo></mrow><mo>=</mo><mrow><mo form="prefix">(</mo><mi>φ</mi><mo>,</mo><mi>ψ</mi><mo
      form="postfix">)</mo></mrow></mrow></math> 以及 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>=</mo><mi>t</mi></mrow></math>
      。为了让 (2.14) 等号成立，我们必须有
    </p>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>φ</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
      form="postfix">)</mo></mrow><mo>+</mo><mi>ψ</mi><mrow><mo form="prefix">(</mo><mi>t</mi><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow><mo>=</mo><mi>c</mi><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo>-</mo><mi>t</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo
      form="postfix">)</mo></mrow><mo form="postfix">)</mo></mrow></mrow></mstyle></math>
    </center>
    <p>
      对几乎任意 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>supp</mi><mo>&ApplyFunction;</mo><mi>f</mi></mrow></math>。因此
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><msub><mi>s</mi><mi>c</mi></msub><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow></mrow></math>对几乎任意
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>supp</mi><mo>&ApplyFunction;</mo><mi>f</mi></mrow></math>，根据step
      2.
    </p>
    <p>
      
    </p>
    <p>
      <div style="display: inline">
        <a id="auto-4"></a>
      </div>
      <h5>后面的章节讨论了蒙日问题的最优传输的构造和证明。实际上，蒙日问的问题的代价函数
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mrow><mo
      form="prefix">|</mo><mi>x</mi><mo form="postfix">|</mo></mrow></mrow></math>
      是最难的一种。严格凸的代价函数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo form="postfix">)</mo></mrow><mo>=</mo><msup><mrow><mo
      form="prefix">|</mo><mi>x</mi><mo>-</mo><mi>y</mi><mo form="postfix">|</mo></mrow><mi>p</mi></msup><mo>,</mo><mi>p</mi><mo>></mo><mn>1</mn></mrow></math>
      要简单一些，这可能是为什么蒙日自己没有解决的原因之一。<span
      style="margin-left: 1em"></span></h5>
    </p>
</body>
