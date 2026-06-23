---
layout: post
title: The founding fathers of optimal transport, Cédric Villani
date: 2026-06-22 22:25:00
description: part of chapter 3 of Optimal Transport, Old and New
tags: OT
categories: math
---

<body>
    <h2 id="auto-1">3<span style="margin-left: 1em"></span>The founding fathers of optimal
    transport<span style="margin-left: 1em"></span></h2>
    <p>
      和数学中的其他研究主题一样，最优传输诞生过多次。第一次发生在十八世纪末，由法国几何学家Gaspard
      Monge提出。
    </p>
    <p>
      Monge 1746年出生在 French Ancient R&eacute;gime.
      由于他出色的技能，被一所陆军训练学校录取。他独立发明了
      descriptive
      geometry(画法几何)，这一方法如此强大的力量使得他在22岁被授予教授，让他的理论保持为军事秘密，仅供高级军官们使用。他后来是法国大革命最热心的战士科学家之一，在多个政权下担任教授，在恐怖统治期间逃过了一个死刑，成为了拿破仑最好的朋友之一。他在巴黎高师和理工学院教书。他的大部分工作投入到几何中。
    </p>
    <p>
      在1781年，他发表了著名的工作之一，M&eacute;moire
      sur la th&eacute;orie des d&eacute;blais et des remblais.
      （d&eacute;blais
      意思是挖掘，remblai意思是填充构造）<span
      style="margin-left: 0.16665em"></span>.<span style="margin-left: 0.16665em"></span>.<span style="margin-left: 0.16665em"></span>.<span style="margin-left: 0.16665em"></span>我们想最小化总的传输代价。Monge
      假设一单位质量的传输代价由质量和距离的乘积给出。
    </p>
    <p>
      今天法国巴黎有一条Monge
      街，
      人们在那里可以找到一家优秀的面包店叫作
      Le Boulanger de Monge.
      为了承认这一点，并展示Monge的问题可以如何从经济学角度叙述，我将如下叙述这个问题。考虑一大群烘焙店，生产面包，需在每天早上运到咖啡店供消费者食用。每个咖啡店要消耗的面包数提前知道，可以被建模为特定空间上的概率测度（有一个&ldquo;生产密度&rdquo;和&ldquo;消耗密度&rdquo;）。问题是每单位的面包应该去哪里来最小化传输代价。所以Monge的问题其实是寻找一个
      optimal coupling;
      更精确地说，他在寻找一个
      deterministic optimal coupling。
    </p>
    <p>
      Monge在三维空间中连续的质量分布上研究了他的问题。在他优美的几何直觉指引下，他作出重要观察发现传输需要沿着直线进行而且会和一族曲面正交。这个研究引领他发现了曲率的线，一个本身对曲面的几何就是重要贡献的概念。他的想法后来由
      Charles Dupin和Paul
      Appell继续发展。在现代数学的标准下，这些论点都是有缺陷的，但显然值得用现代工具重新审视这些问题。
    </p>
    <p>
      在很久之后，Monge的问题被俄罗斯数学家
      Leonid Vitaliyevich Kantorovich
      重新发现。1912年出生的
      Kantorovich
      也是一个非常有天分的数学家，在18岁时就被认可为一流的研究者，并在跟Monge
      一样的年纪获得教授职位。他在数学的很多领域研究，尤其在经济学的应用以及后来的理论计算机科学上有强烈兴趣。在1938年，一个实验室咨询他一个特定优化问题的解，他发现这是在不同经济领域中都出现的一整类线性规划问题的代表。在这个发现的驱动下，他发展了线性规划的工具，后来在经济学中变得著名。他的最重要的部分工作的发表被延后了，因为苏联当局对关于经济学的研究成果的泄密处理得非常谨慎。事实上（这和Monge是另一个共同点）很多年Kantorovich被禁止公开讨论一些他的主要发现。最终他的工作变得广为人知，然后他在1975年和
      Tjalling Koopmans
      一起获得诺贝尔经济学奖，&ldquo;for
      their contributions to the theory of optimum allocation of
      resources&rdquo;. 
    </p>
    <p>
      （Kantorovich独立创立了线性规划方法，提出了乘子，Koopmans在经济学体系中赋予了具体的含义，即著名的&ldquo;影子价格&rdquo;。两人的研究结合在一起，确立了如何将稀缺资源进行最优分配的数学与经济学基础。）
    </p>
    <p>
      对我们直接感兴趣的部分，即optimal
      coupling 来说，Kantorovich
      通过泛函分析工具，描述和证明了，一个对偶定理，其在之后还会起到重要作用。他还推导了概率测度之间的一个方便的距离的记号：两个测度之间的距离应该是最优传输代价。概率测度之间的这个距离今天被叫做
      Kantorovich-Rubinstein
      距离，且已经被证明尤其灵活有用。
    </p>
    <p>
      在作出主要成果仅过了几年之后，Kantorovich
      和Monge的工作建立起了联系。从此，optimal
      coupling 的问题被叫做
      Monge-Kantorovich 问题。
    </p>
    <p>
      在二十世纪的后半部分，最优coupling的技巧和
      Kantorovich-Rubinstein
      距离（今天通常被叫做
      Wasserstein
      距离，或其他名称）被统计学家和概率学家使用。&ldquo;基&rdquo;空间可以是有限维，或是无穷维：例如，在路径空间的概率测度上给出有趣的距离的定义。七十年代值得注意的贡献有
      Roland
      Dobrushin，用这样的距离来研究粒子系统；Hiroshi
      Tananka，用它们研究 Boltzmann
      方程的简单变体的时变行为。八十年代中期，这方面的专家，像
      Svetlozar Rachev 或 Ludger Ruschendorf，
      拥有最优传输相关的一大批想法，工具，技巧和应用。
    </p>
    <p>
      在那期间，reparameterization techniques
      (change of variables
      的另一个名字)在被很多研究人员用在研究体积或积分的不等式上。直到后来人们才意识到最优传输往往提供了有用的重参数化。
    </p>
    <p>
      八十年代末，三个方向几乎同时独立出现，彻底改变了最优传输的整个面貌。
    </p>
    <p>
      一个是 John Mather's 的 Lagrangian dynamical
      systems上的工作。最小化作用量的曲线是动力系统中的基本重要对象，而构造满足特定数量条件的封闭最小化作用量的曲线是一个经典问题。在八十年代末之前，Mather
      发现在研究最小作用量曲线之外，相空间的最小作用量<i>静态</i>测度非常方便。Mather的测度是最小作用量曲线的一个推广，它们解决了一个变分问题，实际上是一个
      Monge-Kantorovich
      问题。在拉格朗日量的一些条件下，Mather证明了一个值得称道的结果（大致来说）是特定最小作用量测度自动集中在Lipschitgz图上。我们在第8章将会明白，这个问题和确定性最优coupling的构造密切相关。
    </p>
    <p>
      第二个研究方向来自
      Yann
      Brenier的工作。在研究不可压缩流体力学的问题时，Brenier需要构造一个算子，其可以在开集的保测度映射上像投影一样作用（在概率语言下，保测度映射是Lebesgue测度到自身的确定性coupling）。他明白他可以通过引入一个最优coupling来实现这一点：If
      \(u\) is the map for which one wants to compute the projection,
      introduce a coupling of the Lebesgue measure \(\mathcal{L}\) with
      \(u_{\#} \mathcal{L}\).
      这一研究揭示了一个人们没有预料到的最优传输和流体力学间的联系；同时，在指出和Monge-Ampere方程之间的联系后，Brenier吸引到了PDE领域研究的注意。
    </p>
    <p>
      第三个研究方向，也是最令人吃惊的，来自数学之外。Mike
      Cullen was part of a group of meteorologists with a well-developed
      mathematical taste, working on semi-geostrophic equations, sused in
      meteorology for the modeling of atmospheric fronts. Cullen and his
      collaborators showd that a certain famous change of unknown due to Brian
      Hoskins could be re-interpreted in terms of an optimal coupling problem,
      and they identified the minimizaiton property as a stability condition.
      A striking outcome of the work was that optimal transport could arise
      naturally in partial differential equations which seemed to have nothing
      to do with it.
    </p>
    <p>
      All three contributions emphasized (in their respective domain) that
      <i>important information can be gained by a qualitative description of
      optimal transport.</i> These new directions of research attracted
      various mathematicians (among the first, Luis Caffarelli, Craig Evans,
      Wilfrid Gangbo, Robert McCann, and others), who worked on a better
      description of the structure of optimal transport and found other
      applications.
    </p>
    <p>
      An important conceptual step was accomplished by Felix Otto, who
      discovered an appealing formalism introducing a <i>differential</i>
      point of view in optimal transport theory. This opened the way to a more
      geometric description of the space of probability measures, and
      connected optimal transport to the theory of diffusion equations, thus
      leading to a rich interplay of geometry, functional analysis and partial
      differential equations.
    </p>
    <p>
      今天（2008～2009）最优传输已经成为一个蓬勃发展的industry，包括许多研究者和许多趋势。除了
      meterology，流体力学和扩散方程，它也被应用到多样的领域，比如沙堆坍塌，图像匹配，网络和反射天线的设计。我的书，Topics
      in Optimal
      Transport，大概在2000到2003年之间写就，是第一次尝试提供一个现代的完整理论视图。在那之后，这个领域发展的比我想象的更快，且它从没有如今天这么活跃。
    </p>
    <p>
      （事实上，可能活跃到了今天2026年，尤其在机器学习领域OT理论和应用还在发展。）
    </p>
  </body>
