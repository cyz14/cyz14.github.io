---
layout: post
title: Minkowski-Type Theorems and Least-Squares Partitioning, Aurenhammer, Hoffmann and Aronov
date: 2026-06-06 00:00:00
description: notes of this paper
tags: least-squares, Minkowski
categories: math
---

<body>
    <h3 id="auto-1">1<span style="margin-left: 1em"></span>Introduction and Statement of Results<span
    style="margin-left: 1em"></span></h3>
    <p>
      考虑 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math> 个点（或叫作站点
      sites）的集合 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>,
      位于欧氏平面上。<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      诱导了平面的 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math>
      个多边形区域的一个自然的划分。每个点 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>
      的区域 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>reg</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>，包含所有到
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>s</mi></math> 的距离比其他 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></math>
      个点都近的点 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math>。这个划分被人们称为
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math> 的 Voronoi
      diagram。如果我们固定平面上 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>m</mi></math>
      个点的集合 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>， 这个集合被 <math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math> 的 Voronoi diagram
      划分为子集。更精确地说，Voronoi diagram
      定义了一个分配函数 (assignment function) <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>:</mo><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>，即
      
    </p>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>A</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow><mo>=</mo><mi>s</mi><mo>⟺</mo><mi>x</mi><mo>∈</mo><mrow><mtext>reg</mtext></mrow><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></mstyle></math>
    </center>
    <p>
      等价于说, <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mo>=</mo><mi>X</mi><mo>∩</mo><mtext>reg</mtext><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>,
      对所有 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>。
      可以观察到这个分配 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math>
      有下面的最优性：它最小化所有顶点和被分配的点之间的距离的和，在所有<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>的分配函数中。
      
    </p>
    <p>
      给定 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math> 和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>，我们希望能够通过调整距离函数改变分配。为了实现，我们附加一组
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mo>=</mo><mrow><mo form="prefix">{</mo><mi>w</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mo>|</mo><mi>s</mi><mo>∈</mo><mi>S</mi><mo
      form="postfix">}</mo></mrow></mrow></math> 
      实数的权重给站点然后把欧氏距离 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>δ</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>
      替换为 power function 
    </p>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><msub><mrow><mtext>pow</mtext></mrow><mi>W</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>s</mi><mo>)</mo></mrow><mo>=</mo><msup><mi>δ</mi><mn>2</mn></msup><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>s</mi><mo>)</mo></mrow><mo>-</mo><mi>w</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mi>.</mi></mrow></mstyle></math>
    </center>
    <p>
      我们得到的平面的划分被叫做 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      在权重 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>W</mi></math> 下的 <em>power
      diagram</em>&#x3002; 
    </p>
    <p>
      感兴趣的读者可以看综述 {% cite Aurenhammer-voronoi-survey-1991 %} 了解 Voronoi-type diagrams 和 power diagrams
      的一般性质。每个区域仍然是凸的，且随着权重减少（扩大）面积缩小（变大）。我们获得了一个分配函数:
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>A</mi><mi>W</mi></msub><mo>:</mo><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>
      现在很明显依赖特定的权重选取。 
    </p>
    <p>
      上面的概念以自然的方式可以扩展到任意维度。我们介绍下面的一般结论
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">1</class>. </strong>令 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math> 为 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi></math>维欧氏空间
      <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mi>d</mi></msup></math> 中的 <math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math> 个站点和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>m</mi></math>
      个点。对于站点的任意整数容量 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>
      满足 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>&Sum;</mo><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></msub><mi>c</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mo>=</mo><mi>m</mi></mrow></math>，存在一组权重
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>W</mi></math> 使得对所有 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>，满足
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo form="prefix">|</mo><msubsup><mi>A</mi><mi>W</mi><mrow><mo>-</mo><mn>1</mn></mrow></msubsup><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mo
      form="postfix">|</mo></mrow><mo>=</mo><mi>c</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>&#x3002;
      
    </p>
    <p>
      换句话说，总存在一个 power diagram 的区域分解一个 <math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi></math> 维有限点集 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>
      为指定大小的
      clusters，不论这些站点被选在哪里。更一般地，令 <math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>A</mi><mi>W</mi></msub><mo>:</mo><msup><mi>𝔼</mi><mi>d</mi></msup><mo>→</mo><mi>S</mi></mrow></math>
      为一个 power diagram 诱导的分配函数，可以看作把整个
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi></math> 维空间的点到 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      的站点的映射。取 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>A</mi><mi>W</mi><mrow><mo>-</mo><mn>1</mn></mrow></msubsup><mo>=</mo><mi>r</mi><mo>&InvisibleTimes;</mo><mi>e</mi><mo>&InvisibleTimes;</mo><msub><mi>g</mi><mi>W</mi></msub><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>，<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math> 带权重 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>W</mi></math>
      的站点 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>s</mi></math> 的区域。 
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">2</class>. </strong>令 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ϱ</mi></math>
      为某个 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mi>d</mi></msup></math>
      上的概率分布，在 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow><mi>d</mi></msup></math>
      外为 0 ，令 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mrow><mo>(</mo><mi>X</mi><mo>)</mo></mrow><mo>=</mo><msub><mo>&Integral;</mo><mi>X</mi></msub><mi>ϱ</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><mi>d</mi><mo>&InvisibleTimes;</mo><mi>x</mi></mrow></math>
      表示 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>⊂</mo><msup><mi>𝔼</mi><mi>d</mi></msup></mrow></math>
      的基于 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ϱ</mi></math>
      的测度。对任意容量函数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>:</mo><mi>S</mi><mo>→</mo><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow></mrow></math>
      满足 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>&Sum;</mo><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></msub><mi>c</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mo>=</mo><mn>1</mn></mrow></math>，存在一组
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>W</mi></math> 使得对所有<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>，有
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mrow><mo>(</mo><msub><mtext>reg</mtext><mi>W</mi></msub><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow><mo>)</mo></mrow><mo>=</mo><mi>c</mi><mrow><mo>(</mo><mi>s</mi><mo>)</mo></mrow></mrow></math>.
    </p>
    <p>
      通过取 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ϱ</mi></math> 为 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>]</mo></mrow><mi>d</mi></msup></math>
      均匀分布，我们得到 
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>推论 <class style="font-style: normal">1</class>. </strong><a id="cor:power-diagram"></a>对任意
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math> 个站点，存在一个 power
      diagram把单位超立方体分成 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math>
      个指定体积的凸多边形区域。
    </p>
    <p>
      这看起来很惊喜，因为站点的位置决定了划分区域的面的法向。推论
      <a href="#cor:power-diagram">1</a> 跟凸多面体的Minkowski problem有关（见 {% cite grunbaum_convex_2003 %}
      ），根据我们的需要，可以被描述为：选任意 <math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math> 个欧氏空间 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mrow><mi>d</mi><mo>+</mo><mn>1</mn></mrow></msup></math>
      中的没有互相平行的向量。存在一个 <math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mrow><mi>d</mi><mo>+</mo><mn>1</mn></mrow></msup></math>
      中的凸多面体有 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></math>个面，每个面垂直于对应的法向量且其
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>d</mi></math>维体积等于其长度。<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mi>d</mi></msup></math>中的power
      diagram 恰是<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mrow><mi>d</mi><mo>+</mo><mn>1</mn></mrow></msup></math>中无界凸多面体的投影 {% cite Aurenhammer-1987 %}，可以被看成第<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo form="prefix">(</mo><mi>n</mi><mo>+</mo><mn>1</mn><mo
      form="postfix">)</mo></mrow></math>个面位于投影方向的无穷远处，Corollary
      <a href="#cor:power-diagram">1</a> 其实是无界多面体的Minkowski定理的推广。
    </p>
    <p>
      我们会看到，分配函数<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>A</mi><mi>W</mi></msub></math>有很好的性质，最小化站点和被分配的点之间的距离之和。这样的一个分配叫作
      <i>least-squares assignment</i> subject to capacity
      function。更一般的，下面的定理显示
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>定理 <class style="font-style: normal">3</class>. </strong>令 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      为一组<math xmlns="http://www.w3.org/1998/Math/MathML"><msup><mi>𝔼</mi><mi>d</mi></msup></math>中的有限站点。<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math> 的 power diagram
      诱导的任意分配是一个 least-squares assignment，
      服从容量约束。反过来，<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      的一个最小平方分配，满足任意容量约束的，存在且可以被一个
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math> 的power diagram实现。
    </p>
    <p>
      为了展示这个概念&ldquo;限制最小平方分配&rdquo;<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>:</mo><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>的作用，我们介绍一些<math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>L</mi></math>在<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>是有限集合时的性质。
    </p>
    <ol>
      <li>
        <p>
          诱导得到的集群 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>L</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mrow><mo
          form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>,</mo><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>
          是彼此凸包不相交的。如果我们选择最小化一个不同的函数，比如到站点的欧氏距离的和，而不是平方距离的和，最优分配不会保证有这个不相交性质。clusters凸包的不相交性是我们希望的，因为比如它可以让新的点的分类变得容易。
        </p>
      </li>
      <li>
        <p>
          在 section
          5中我们会展示解决一个特定的传输问题等价于找到预定大小的最小平方集群。
        </p>
      </li>
      <li>
        <p>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>L</mi></math> 在平移和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
          的缩放下不变。这个性质有用，比如为了找到两个
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math> 个顶点的集合 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
          和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>之间最好的最小平方匹配，如果平移和
          <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
          的缩放被允许的话。它保证最小平方分配 (<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo
          form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow><mo>=</mo><mn>1</mn></mrow></math>
          对所有 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>)就是最优匹配下的双射。但是，给定双射，计算平移和缩放系数很简单。最小平方匹配和最小平方指定(fitting)之间的联系在
          section 5 讨论。
        </p>
      </li>
    </ol>
    <p>
      利用power diagram
      的机制作者提出了两种算法计算限制最小平方分配。第一个对有限点集
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>。从 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>S</mi></math>
      的Voronoi diagram出发，所有 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math>
      个权重为 0，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mi>∅</mi></mrow></math>，然后逐步把
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>m</mi></math> 个点插入 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math>，每一步调整权重使得容量不超过限制。在平面上，时间复杂度是
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>O</mi><mrow><mo form="prefix">(</mo><msup><mi>n</mi><mn>2</mn></msup><mi>m</mi><mo>&ApplyFunction;</mo><mi>log</mi><mo>&ApplyFunction;</mo><mi>m</mi><mo>+</mo><mi>n</mi><mo>&ApplyFunction;</mo><mi>m</mi><mo>&ApplyFunction;</mo><mi>log</mi><msup><mo>&ApplyFunction;</mo><mn>2</mn></msup><mi>m</mi><mo
      form="postfix">)</mo></mrow></mrow></math> 和最优空间复杂度 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>O</mi><mrow><mo
      form="prefix">(</mo><mi>m</mi><mo form="postfix">)</mo></mrow></mrow></math>。这改进了之前最优的算法，通过转为最小费用流求解的方法。此外，还要提及
      Tokuyama 和 Nakano
      的随机化算法，比本文的算法更加一般，适合任意代价函数。
    </p>
    <p>
      第二个算法对有限和连续版本的问题都可用。它依赖于以下有趣的事实：找一个权重向量
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>W</mi></math> 使得 <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>A</mi><mi>W</mi></msub></math>
      在容量约束下最优等价于找到一个权重空间上 <math
      xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math>元凹函数的最大值。作者提出了梯度方法迭代优化权重向量。这个方法有超线性收敛速度只要概率分布
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>ϱ</mi></math>
      是连续的。空间复杂度是最优的 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>O</mi><mrow><mo
      form="prefix">(</mo><mi>n</mi><mo form="postfix">)</mo></mrow></mrow></math>&#x3002;
    </p>
    <p>
      在有限情形，n元函数是分片线性的，最大值是满维度的。找到一个最大值点变成线性规划问题，但是限制条件的数量是
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math>
      的指数。迭代算法仍然可用；可以保证在有限步后停止。
    </p>
    <p>
      
    </p>
    <h3 id="auto-2">2<span style="margin-left: 1em"></span>Proof of Theorem 3<span style="margin-left: 1em"></span></h3>
    <p>
      略。
    </p>
    <h3 id="auto-3">3<span style="margin-left: 1em"></span>Computing the Weights<span style="margin-left: 1em"></span></h3>
    <p>
      略。
    </p>
    <h3 id="auto-4">4<span style="margin-left: 1em"></span>An Iterative Approach<span style="margin-left: 1em"></span></h3>
    <p>
      略.
    </p>
    <p>
      作者证明了迭代在有限步后收敛，并在二维实现了该算法。在100个站点和1000个点均匀分布在矩形区域的测试数据，基本都不到10步就收敛了。还提到，在确定一个
      <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>W</mi><mi>k</mi></msub></math> 对 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>W</mi><mo>∗</mo></msup><mo>&ApplyFunction;</mo></mrow></math>的好的近似后，插入算法应该从
      <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>W</mi><mi>k</mi></msub></math> 初始化而不是
      0.
    </p>
    <h3 id="auto-5">5<span style="margin-left: 1em"></span>Some Applications<span style="margin-left: 1em"></span></h3>
    <p>
      容量限制的最小平方分配，作为一个太自然的概念，有若干应用。
    </p>
    <p>
      对于 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>⊂</mo><mi>X</mi><mo>,</mo><mi>s</mi><mo>∈</mo><mi>S</mi></mrow></math>，定义
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math>相对 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>s</mi></math> 的
      variance 为 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>&Sum;</mo><mrow><mi>x</mi><mo>∈</mo><mi>Y</mi></mrow></msub><msup><mi>δ</mi><mn>2</mn></msup><mrow><mo
      form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>s</mi><mo form="postfix">)</mo></mrow></mrow></math>。那么一个限制最小平方分配
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>:</mo><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>
      恰是一个有指定容量分类且分类variances最小。除了上述最优性，这些分类有一个重要性质是凸包的不相交。
    </p>
    <p>
      如果我们定义 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Y</mi></math> 相对 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>s</mi></math>
      的 profit（利润）为 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>&Sum;</mo><mrow><mi>x</mi><mo>∈</mo><mi>Y</mi></mrow></msub><mrow><mo
      form="prefix">&LeftAngleBracket;</mo><mi>x</mi><mo>,</mo><mi>s</mi><mo form="postfix">&RightAngleBracket;</mo></mrow></mrow></math>，那么
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>L</mi></math>
      显然最大化各分类利润的和。（这对应最优传输的经济学视角）
    </p>
    <p>
      下一个应用利用了容量限制最小平方分配的平移和缩放不变性。
    </p>
    <p style="margin-top: 1em; margin-bottom: 1em">
      <strong>注意 <class style="font-style: normal">1</class>. </strong>令 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo>∈</mo><msup><mi>ℝ</mi><mo>+</mo></msup><mo>,</mo><mi>τ</mi><mo>∈</mo><msup><mi>𝔼</mi><mi>d</mi></msup></mrow></math>，考虑一个最小平方分配
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>:</mo><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>
      和容量限制 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>c</mi></math>。那么<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>L</mi></math>
      也是一个 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>X</mi></math> 到 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mi>S</mi><mo>+</mo><mi>τ</mi></mrow></math>
      在相同容量限制下的最优分配。
    </p>
    <p>
      证明略。
    </p>
    <p>
      假设 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi><mo>,</mo><mi>X</mi></mrow></math>
      有相同基数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>n</mi></math>，考虑 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>:</mo><mi>X</mi><mo>→</mo><mi>S</mi></mrow></math>
      满足 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mrow><mo form="prefix">(</mo><mi>s</mi><mo
      form="postfix">)</mo></mrow><mo>=</mo><mn>1</mn></mrow></math>。定义
      least-squares fitting 为 least-squares matching <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>L</mi><mo>∗</mo></msub></math>
      使得 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>L</mi><mo>∗</mo></msub><mo>:</mo><mi>X</mi><mo>→</mo><mi>σ</mi><mi>S</mi><mo>+</mo><mi>τ</mi></mrow></math>
      的值对所有缩放系数 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>σ</mi></math>
      和平移向量 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>τ</mi></math> 来说最小。上面的
      Remark告诉我们 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>L</mi><mo>∗</mo><mrow><mo>-</mo><mn>1</mn></mrow></msubsup><mrow><mo
      form="prefix">(</mo><mi>σ</mi><mi>s</mi><mo>+</mo><mi>τ</mi><mo form="postfix">)</mo></mrow><mo>=</mo><msup><mi>L</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mrow><mo
      form="prefix">(</mo><mi>s</mi><mo form="postfix">)</mo></mrow></mrow></math>。这说明，计算
      least-squares fitting 时，我们可以先计算和固定 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>L</mi></math>然后计算最优的
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>σ</mi></math> 和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>τ</mi></math>&#x3002;
    </p>
    <p>
      后一个任务实际上是简单的，当 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>&ApplyFunction;</mo></mrow></math>固定时。令
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi><mo>=</mo><mrow><mo form="prefix">{</mo><msub><mi>s</mi><mn>1</mn></msub><mo>,</mo><mi>…</mi><mo>,</mo><msub><mi>s</mi><mi>n</mi></msub><mo
      form="postfix">}</mo></mrow></mrow></math>，<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>L</mi><mrow><mo>-</mo><mn>1</mn></mrow></msup><mrow><mo
      form="prefix">(</mo><msub><mi>s</mi><mi>i</mi></msub><mo form="postfix">)</mo></mrow><mo>=</mo><msub><mi>x</mi><mi>i</mi></msub></mrow></math>。我们希望找到
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>σ</mi></math> 和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>τ</mi><mo>=</mo><mrow><mo
      form="prefix">(</mo><msub><mi>τ</mi><mn>1</mn></msub><mo>,</mo><mi>…</mi><mo>,</mo><msub><mi>τ</mi><mi>d</mi></msub><mo
      form="postfix">)</mo></mrow></mrow></math> 使得
    </p>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>Q</mi><mrow><mo form="prefix">(</mo><mi>σ</mi><mo>,</mo><mi>τ</mi><mo
      form="postfix">)</mo></mrow><mo>=</mo><munderover><mo>&Sum;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><mi>δ</mi><mn>2</mn></msup><mrow><mo
      form="prefix">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>,</mo><mi>σ</mi><msub><mi>s</mi><mi>i</mi></msub><mo>+</mo><mi>τ</mi><mo
      form="postfix">)</mo></mrow></mrow></mstyle></math>
    </center>
    <p>
      最小。取偏导数 <math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>Q</mi><mi>σ</mi></msub></math>
      和 <math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>Q</mi><msub><mi>σ</mi><mi>j</mi></msub></msub><mo>,</mo><mn>1</mn><mo>≤</mo><mi>j</mi><mo>≤</mo><mi>d</mi></mrow></math>，为0，求得最小值取在
    </p>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>σ</mi><mo>=</mo><mfrac><mrow><mi>a</mi><mo>&ApplyFunction;</mo><mi>n</mi><mo>-</mo><mrow><mo
      form="prefix">&LeftAngleBracket;</mo><mi>α</mi><mo>,</mo><mi>β</mi><mo form="postfix">&RightAngleBracket;</mo></mrow></mrow><mrow><mi>b</mi><mo>&ApplyFunction;</mo><mi>n</mi><mo>-</mo><mrow><mo
      form="prefix">&LeftAngleBracket;</mo><mi>β</mi><mo>,</mo><mi>β</mi><mo form="postfix">&RightAngleBracket;</mo></mrow></mrow></mfrac><mo>,</mo><mi>τ</mi><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><mrow><mo
      form="prefix">(</mo><mi>α</mi><mo>-</mo><mi>σ</mi><mi>β</mi><mo form="postfix">)</mo></mrow></mrow></mstyle></math>
    </center>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>a</mi><mo>=</mo><mo>&Sum;</mo><mrow><mo
      form="prefix">&LeftAngleBracket;</mo><msub><mi>x</mi><mi>i</mi></msub><mo>,</mo><msub><mi>s</mi><mi>i</mi></msub><mo
      form="postfix">&RightAngleBracket;</mo></mrow><mo>,</mo><mi>b</mi><mo>=</mo><mo>&Sum;</mo><mrow><mo
      form="prefix">&LeftAngleBracket;</mo><msub><mi>s</mi><mi>i</mi></msub><mo>,</mo><msub><mi>s</mi><mi>i</mi></msub><mo
      form="postfix">&RightAngleBracket;</mo></mrow><mo>,</mo></mrow></mstyle></math>
    </center>
    <center>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><mstyle displaystyle="true"><mrow><mi>α</mi><mo>=</mo><mo>&Sum;</mo><msub><mi>x</mi><mi>i</mi></msub><mo>,</mo><mi>β</mi><mo>=</mo><mo>&Sum;</mo><msub><mi>s</mi><mi>i</mi></msub><mi>.</mi></mrow></mstyle></math>
    </center>
    <p>
      
    </p>
    
<h3 id="auto-1">References<span style="margin-left: 1em"></span></h3>

{% bibliography --cited %}

</body>
