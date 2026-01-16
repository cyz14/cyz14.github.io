---
layout: post
title: how-gauss-predicted-Ceres-orbit-part-2
date: 2023-10-05 22:18:00
description: a translation of Bachelor Thesis of Daniel Bed’at\check{s}, Gauss’ calculation of Ceres’ orbit
tags: translation
categories: personal
chart:
  plotly: true
---

因为文章太长编辑器建议分开发送。这里接着[上一篇](/blog/2023/how-gauss-predicted-Ceres-orbit-part-1)。 上篇原文发在[高斯是如何计算谷神星轨道参数的？（上）](https://blog.csdn.net/cyz14/article/details/132032662)。

## 主要结果

这里我们已经面临最后的任务，去计算 $\tau'$ 时刻太阳和谷神星的距离 $\delta'$。这里还是会反复用到我们前面用到的工具，包括椭圆的极坐标公式，开普勒第二和第三定律，以及式（2.6）的近似版本等。

首先回忆真偏角 $\theta$ 以及椭圆的极坐标公式

$$
r(\theta)=\frac{k}{1+e\cos \theta}
$$

对于三个时刻$\tau,\tau',\tau''$来说我们有三个等式

$$
r = \frac{k}{1 + e \cos \theta}, r' = \frac{k}{1 + e
   \cos \theta'}, r'' = \frac{k}{1 + e \cos \theta''}
$$

两边取倒数我们有

$$
\frac{1}{r} = \frac{1 + e \cos \theta}{k}, \frac{1}{r'} = \frac{1 + e \cos
\theta'}{k}, \frac{1}{r''} = \frac{1 + e \cos \theta''}{k} \tag{2.10}
$$

然后我们分别用 $\sin(\theta''-\theta'),\sin(\theta-\theta''),\sin(\theta'-\theta)$ 乘以上式两边并加起来得到式（2.14），其中左边是（2.11）

$$
\frac{\sin (\theta'' - \theta)}{r} + \frac{\sin (\theta - \theta'')}{r'} +
\frac{\sin (\theta' - \theta)}{r''} = \frac{2 f + 2 f' + 2 f''}{r r' r''} \tag{2.11}
$$

因为有三角形的面积公式 $2 f = r' r'' \sin (\theta'' - \theta'), 2 f' = r r'' \sin (\theta -
\theta''), 2 f'' = r r' \sin (\theta' - \theta)$，在引理4中我们证明了其中一种。
右边是

$$
  \frac{1 + e \cos \theta}{k} \sin (\theta'' - \theta') + \frac{1 + e \cos
  \theta'}{k} \sin (\theta - \theta'') + \frac{1 + e \cos \theta''}{k} \sin
  (\theta' - \theta)\\
  = \frac{1}{k} [\sin (\theta'' - \theta') + \sin (\theta - \theta'') + \sin
  (\theta' - \theta) + \\ + e (\cos \theta \sin (\theta'' - \theta') + \cos \theta' \sin (\theta -
  \theta'') + \cos \theta'' \sin (\theta' - \theta))]
$$

里面e乘的括号中的项可以被证明等于和为0：

$$
\begin{array}{l}
  \cos \theta \sin (\theta'' - \theta') + \cos \theta' \sin (\theta -
  \theta'') + \cos \theta'' \sin (\theta' - \theta)\\
  = \cos \theta (\sin \theta'' \cos \theta' - \sin \theta' \cos \theta'') +
  \cos \theta' (\sin \theta \cos \theta'' - \sin \theta' \cos \theta)\\+ \cos \theta'' (\sin \theta' \cos \theta - \sin \theta \cos \theta') = 0
\end{array}
$$

因此右边等于

$$
\frac{\sin (\theta'' - \theta') + \sin (\theta - \theta'') + \sin (\theta' -
\theta)}{k} \tag{2.12}
$$

这一项可以进一步变形来使用，但要引入一个费解的三角恒等式。

**引理 9** 对所有 $U,V \in \R$，有下列关系

$$
\sin U + \sin V - \sin (U + V) = 4 \sin \left( \frac{U}{2} \right) \sin
\left( \frac{V}{2} \right) \sin \left( \frac{U + V}{2} \right)
$$

证明对$U,V$用到 sin的加法公式以及二倍角公式 $\cos U=1-2\sin^2(U/2),\sin U=2\sin(U/2)\cos(U/2)$。

考虑令 $U=\theta''-\theta', V=\theta'-\theta$，然后 $U+V=\theta''-\theta$，以及由于 $\sin(\theta-\theta'')=-\sin(\theta''-\theta)$，我们可以把（2.12）变形为

$$
\frac{4}{k} \sin \left( \frac{\theta'' - \theta'}{2} \right) \sin \left(
\frac{\theta' - \theta}{2} \right) \sin \left( \frac{\theta'' - \theta}{2}
\right) \tag{2.13}
$$

现在，我们从（2.10）得到的左右两边就构成（2.14）

$$
\frac{2 f + 2 f' + 2 f''}{r r' r''} = \frac{4}{k} \sin \left( \frac{\theta'' - \theta'}{2} \right) \sin \left( \frac{\theta' - \theta}{2} \right) \sin
\left( \frac{\theta'' - \theta}{2} \right) \tag{2.14}
$$

回忆一下我们有面积公式 $-2f'=rr''\sin(\theta''-\theta)$。我们再利用2倍角公式

$$
\sin(\theta''-\theta)=2\sin(\frac{\theta''-\theta}{2})\cos(\frac{\theta''-\theta}{2})
$$

就得到（2.14）其中一项sin的表达式

$$
\sin(\frac{\theta''-\theta}{2})=\frac{\sin(\theta''-\theta)}{2\cos(\frac{\theta''-\theta}{2})}=\frac{-2f'}{2rr''\cos(\frac{\theta''-\theta}{2})}=\frac{-f'}{rr''\cos(\frac{\theta''-\theta}{2})}
$$

替换（2.14）中的这一项，消去 2 和 $rr''$ 得到

$$
\frac{f + f' + f''}{r'} = \frac{2}{k} \sin \left( \frac{\theta'' -
\theta'}{2} \right) \sin \left( \frac{\theta' - \theta}{2} \right) \frac{-
f'}{\cos \left( \frac{\theta'' - \theta}{2} \right)}
$$

乘以$r'$再除以 $f'$ 得到

$$
\frac{f + f' + f''}{f'} = - \frac{2 r'}{k} \sin \left( \frac{\theta'' -
\theta'}{2} \right) \sin \left( \frac{\theta' - \theta}{2} \right)
\frac{1}{\cos \left( \frac{\theta'' - \theta}{2} \right)} \tag{2.15}
$$

现在我们把注意力放到椭圆参数 $k$ 上。我们在前面把椭圆面积表示为 $\pi a^{3/2}\sqrt{k}$，其中 $a$ 表示椭圆轨道的半长轴，也是六个参数之一。我们用开普勒第二定律来比较扫过的轨道面积和对应的时间区间

$$
\frac{\pi a^{3/2}\sqrt{k}}{t_p}=\frac{g}{\tau''-\tau'}=\frac{g''}{\tau'-\tau}
$$

其中 $t_p$ 表示谷神星的周期。因此，我们也可以得到

$$
\frac{\pi^2 a^3 k}{t_p^2} = \frac{g g''}{(\tau'' - \tau') (\tau' - \tau)}
$$

再回忆开普勒第三定律说

$$
\frac{A^3}{T_e^2} = \frac{a^3}{t_p^2}
$$

其中 $A$ 是地球轨道半长轴，$T_e$ 是一年。所以我们就有

$$
\frac{\pi^2 A^3 k}{T_e^2} = \frac{g g''}{(\tau'' - \tau') (\tau' - \tau)}
$$

因此

$$
k = \frac{T_e^2 g g''}{\pi^2 A^3 (\tau'' - \tau') (\tau' - \tau)} \tag{2.16}
$$

为了继续前进，我们定义三个时刻的三个新的变量 $M,M',M''$ 为

$$
M = \frac{2 \pi}{T_e} (\tau - \tau_e), M' = \frac{2 \pi}{T_e} (\tau' -
\tau_e), M'' = \frac{2 \pi}{T_e} (\tau'' - \tau_e)
$$

其中 $\tau_e$ 表示地球最近一次经过近日点 perihelion 的时刻。我们可以把 $M$ 解读为一个理论的角度，也就是假设地球在假想的具有相同周期的圆形轨道上以恒定角速度在 $\tau$ 时刻经过近日点后运行的角度。（实际上，$M$ 通常在天文学中使用，被叫作 mean anomaly）。

注意，这三个值根据定义都是已知的：地球的周期是1年，地球的近日点时间也是有良好记录的，以及时刻 $\tau,\tau',\tau''$ 都来自 Piazzi 的观测数据。更进一步，我们有

$$
\frac{M'' - M'}{\tau'' - \tau'} = \frac{2 \pi}{T_e (\tau'' - \tau')} ((\tau''- \tau_e) - (\tau' - \tau_e)) = \frac{2 \pi}{T_e}
$$

这与我们前面的解读是一致的，这个常数角速度可以用完整的角度除以整个周期得到。
类似的，我们有

$$
\frac{M'-M}{\tau'-\tau}=\frac{2\pi}{T_e}
$$

把两个式子取倒数再相乘我们得到

$$
\frac{T_e^2}{4\pi^2}=\frac{(\tau''-\tau')(\tau'-\tau)}{(M''-M')(M'-M)}
$$

将此式代入（2.16），消去时间差，就得到

$$
k=\frac{4gg''}{A^3(M''-M')(M'-M)} \tag{2.17}
$$

下一步我们需要引入一些小角度的近似。具体来说，

$$
\cos(\frac{\theta''-\theta}{2})\approx 1, \space  rr''\approx (r')^2
$$

这些十分直观。另外，当观测的小段时间内 $\bold{r},\bold{r}''$ 之间不包括近日点和远日点时，总有 $r<r'<r''$ 或 $r>r'>r''$，因此这些误差在相乘时会彼此抵消部分。进一步，我们近似扫过的椭圆区域面积为

$$
g\approx r'r''\sin(\frac{\theta''-\theta'}{2}), g''\approx rr'\sin(\frac{\theta'-\theta}{2})
$$

这在几何上有一定意义（看图2.6）。考虑左边对 $g$ 的近似，式子的右边是右边以 $r'$ 为底，以

$$
r'' \sin(\frac{\theta''-\theta'}{2})
$$

为高的三角形面积的2倍。
![图2.6](https://i-blog.csdnimg.cn/blog_migrate/ae13b1263ba9afabf41d17edf18b5819.png)
且该三角形的面积等于以 $r''$ 为底，高为

$$
r'\sin(\frac{\theta''-\theta'}{2})
$$

的三角形面积。这两个三角形的面积都接近各自的$g$的面积，因此他们的和接近 $g$。
我们写为以下形式

$$
\sin(\frac{\theta''-\theta'}{2})\approx \frac{g}{r'r''}, \text{  } \sin(\frac{\theta'-\theta}{2})\approx \frac{g''}{rr'}
$$

这样我们就有了式（2.15）中的两项 $\sin$，然后我们用1近似其中的 $\cos$，就得到

$$
\frac{f + f' + f''}{f'} \approx - \frac{2 r'}{k} \frac{g}{r' r''}
\frac{g''}{r r'} \approx - \frac{2 g g''}{k (r')^3}
$$

再用式（2.17）替换 $k$，就有

$$
\frac{f + f' + f''}{f'} \approx - \frac{A^3 (M'' - M') (M' - M)}{4 g g''}
\cdot \frac{2 g g''}{(r')^3}
$$

消去面积后我们就得到最终的近似

$$
\frac{f + f' + f''}{f'} \approx - \frac{A^3 (M'' - M') (M' - M)}{2 (r')^3} \tag{2.18}
$$

这是以 $r'$ 和已知量表示的近似。对地球而言用类似的记号我们同样可以得到

$$
\frac{F + F' + F''}{F'} \approx - \frac{A^3 (M'' - M') (M' - M)}{2 (R')^3} \tag{2.19}
$$

将这两式结合起来，我们有

$$
\frac{f + f' + f''}{f'} - \frac{F + F' + F''}{F'} \approx \frac{A^3}{2} (M''- M') (M' - M) \left( \frac{1}{(R')^3} - \frac{1}{(r')^3} \right)
$$

两边乘以 $f'F'$得到

$$
(f + f' + f'') F' - (F + F' + F'') f' \approx f' F' \frac{A^3}{2} (M'' - M')
(M' - M) \left( \frac{1}{(R')^3} - \frac{1}{(r')^3} \right)
$$

化简左边后得到

$$
(f + f'') F' - (F + F'') f' \approx\\ f' F' \frac{A^3}{2} (M'' - M') (M' - M)
\left( \frac{1}{(R')^3} - \frac{1}{(r')^3} \right)  \tag{2.20}
$$

现在我们终于可以回到第一个主要的式（2.6）。我们先重新完整写下

$$
\begin{array}{l}
  (F + F'') f' \delta' \det (\bold{w} | \bold{w}' | \bold{w}'')\\
  = (f'' F - f F'') (D\det (\bold{w} | \bold{W} |
  \bold{w}'') - D'' \det (\bold{w} | \bold{W}'' |
  \bold{w}''))\\ + ((f + f'') F' - (F + F'') f') D' \det (\bold{w} | \bold{W}' |
  \bold{w}'')
\end{array}
$$

然后回忆之前结论里我们提到右边第一项接近0，因此我们有近似版本

$$
 (F + F'') f' \delta' \det (\bold{w} | \bold{w}' | \bold{w}'') \approx ((f + f'') F' - (F + F'') f') D' \det (\bold{w} | \bold{W}' |
  \bold{w}'')
$$

然后我们最终把经过乏味的计算得到的式（2.20）代入上式，得到

$$
 (F + F'') f' \delta' \det (\bold{w} | \bold{w}' | \bold{w}'') \\
\approx f' F' \frac{A^3}{2} (M'' - M') (M' - M) \left( \frac{1}{(R')^3} -
\frac{1}{(r')^3} \right) D' \det \left( \underset{}{\bold{w}} |
\bold{W}' | \bold{w}'' \right)
$$

注意我们可以消去 $f'$。然后我们把注意力转到 $F,F',F''$。我们可以直接计算这些值（跟地球轨道有关所以我们认为可以计算），不过为了简单，Gauss 用了另一个近似 $F+F''\approx -F'$。这意味着我们选择忽略三个时刻地球所在的三点 $P,P',P''$ 构成的小三角形的面积 $F+F'+F''$，如图2.7中所示。因此我们用 $-(F+F'')$ 替换右边的 $F'$，就可以消去 $F'$，得到

$$
-\delta' \det (\bold{w} | \bold{w}' | \bold{w}'') \approx
\frac{A^3}{2} (M'' - M') (M' - M) \left( \frac{1}{(R')^3} - \frac{1}{(r')^3}
\right) D' \det (\bold{w} | \bold{W}' | \bold{w}'')
$$

![图2.7](https://i-blog.csdnimg.cn/blog_migrate/bd339b8ac341bab57ba52c852a298281.png)
现在我们回忆坐标系的定义，尤其是 $xy$ 平面，我们有 $R'=D'$。移动一些因子我们可以得到

$$
-\frac{\det (\bold{w} | \bold{w}' | \bold{w}'')}{\det (\bold{w} | \bold{W}' | \bold{w}'')} \cdot \frac{2}{(M'' - M')
(M' - M)} \approx \left( 1 - \frac{(R')^3}{(r')^3} \right) \frac{R'}{\delta'} \tag{2.21}
$$

我们现在很接近结束了。（终于！）但是，在我们到达高潮前，我们需要一个最终的辅助引理。
**引理 10** 下面关于 $R',r',\delta'$和观测值的关系成立

$$
\frac{R'}{r'} = \frac{R'}{\delta'} \left( 1 + \tan^2 \beta' + \left(
   \frac{R'}{\delta'} \right)^2 + 2 \frac{R'}{\delta'} \cos (\lambda' - L')
   \right)^{- 1 / 2}
$$

证明：这个等式是定义和一些三角函数的结果。在我们开始必要的记号前，我们注意这里用到的记号都是时刻 $\tau'$ 的。为了避免误解，我们仍然用单引号标记新的变量，尽管我们已经不需要其他两个版本的变量。
![图2.8](https://i-blog.csdnimg.cn/blog_migrate/ad0f7735b392c091a539048f1f3da5e1.png)

首先，令 $C',E',S'$表示 $\tau'$ 时的Ceres，地球和太阳的位置。令 $P'$ 表示 $C'$ 在 $x'y'$ 平面上的投影，记 $u'=|S'P'|$。我们有两个三角形 $E'P'C'，S'P'C'$，都有一个直角在 $P'$ 点和高 $P'C'$。在第一个三角形中，显然有 $|P'C'|=\delta'\tan\beta'$。第二个三角形中，根据勾股定理，

$$
(u')^2=(r')^2-(\delta')^2\tan^2\beta' \tag{2.22}
$$

然后我们考虑 $x'y'$ 平面内的三角形 $S'E'P'$。$E'$ 处的角为 $\pi -\alpha$，其中 $\alpha=|\lambda'-L'|$ 当 $|\lambda'-L'|<\pi$，否则 $\alpha=2\pi-|\lambda'-L'|$ 。任一种情况下，我们都有 $\cos\alpha=\cos|\lambda'-L'|=\cos(\lambda'-L')$， 这样我们可以用余弦定理推导出

$$
\begin{array}{ll}
(u')^2 &= (R')^2 + (\delta')^2 - 2 R' \delta' \cos (\pi - \alpha)\\
          &= (R')^2 + (\delta')^2 - 2 R' \delta' \cos (\lambda' - L')
\end{array}
$$

用式（2.22）中的 $u'$替换，两边再同时加上 $(\delta')^2\tan^2\beta'$ 可以得到

$$
(r')^2 = (R')^2 + (\delta')^2 + (\delta')^2 \tan^2 \beta' + 2 R' \delta' \cos
(\lambda' - L')
$$

为了结束这部分，我们把上式两边同时乘以$(R')^2$再除以$(r')^2(\delta')^2$得到

$$
\left( \frac{R'}{\delta'} \right)^2 = \left( \frac{R'}{r'} \right)^2 \left(
\left( \frac{R'}{\delta'} \right)^2 + 1 + \tan^2 \beta' + 2 \frac{R'}{\delta'}
\cos (\lambda' - L') \right)
$$

两边除以右边的大的括号中的项再开方就得到了我们的引理。

现在我们几乎有了所有需要的东西来结束这一切。接下来需要展开式（2.21）中的行列式。我们在引理6中已经计算了 $\det(\bold{w}|\bold{w}'|\bold{w''})$

$$
\det(\bold{w}|\bold{w}'|\bold{w''})=\tan\beta\sin(\lambda''-\lambda')-\tan\beta'\sin(\lambda''-\lambda)+\tan\beta''\sin(\lambda'-\lambda)
$$

再回忆一次 $B'=0$，所以 $\tan B'=0$，第二个行列式等于

$$
\det (\bold{w} | \bold{W}' | \bold{w}'')= \left|\begin{array}{c}
  \cos \lambda \quad \cos L' \quad \cos \lambda''\\
  \sin \lambda \quad \sin L' \quad \sin \lambda''\\
  \tan \beta \quad \tan B' \quad \tan \beta''
\end{array}\right| \\= \tan \beta (\cos L' \sin \lambda'' - \cos \lambda'' \sin
L') +\\
\tan \beta'' (\cos \lambda \sin L' - \cos L' \sin \lambda) \\
= \tan \beta \sin(\lambda'' - L) + \tan \beta'' \sin (L' - \lambda)
$$

代入式（2.21），我们就得到最终的近似式

$$
\left( 1 - \left( \frac{R'}{r'} \right)^3 \right) \frac{R'}{\delta'} \approx\\
-2 \frac{\tan \beta \sin (\lambda'' - \lambda') - \tan \beta' \sin
(\lambda'' - \lambda) + \tan \beta'' \sin (\lambda' - \lambda)}{(M'' - M') (M' - M) (\tan \beta \sin (\lambda'' - L') + \tan \beta'' \sin (L' - \lambda))}
$$

注意上式中右边的变量全都来自Piazzi的观测数据或可以从地球完备记录的轨道信息得出。使用引理10中的替换，我们得到一个只有一个未知量$R'/\delta'$的方程。这个方程相当复杂，但是可以用数值方法求解。因为 $R'$ 的值已知，我们就可以直接从数值解得到 $\delta'$ 的值了。这样我们就有了前面计算 $\bold{r},\bold{r''}$ 所需要的。有必要的话，我们可以根据

$$
\bold{r}'=(X',Y',Z')^T+\delta'\bold{w}'
$$

来计算 $r'$。这样，我们认为就计算完了。

这里我们主要是根据第三手资料，也就是 $\text{Daniel Bed' at}\check{s}$ 参考 Gauss 1809和 Teets and Whitehead, 1999 完成的本科学位论文，回顾了高斯对谷神星轨道的计算过程。从这些过程我们可以发现，高斯没有用到或者用的不是我们今天所理解的形式上的最小二乘法，计算过程中也就没有直接考虑观测数据的误差的分布，只用三个观测数据，就完成了主要的计算。

## Gauss 的改进

$\text{Daniel Bed' at}\check{s}$ 在论文中说了，Teets 和 Whitehead 的文章到这里就结束了，不过Gauss在自己最初的1809年的文章中，还提供了如何改进这些估计的注释。他提供了两种方式：可以用更加精确的近似，但他发现十分困难因此他自己立刻就放弃了，或是用稍微修改的输入重复这一过程。如 Abdulle and Wanner [2003] 提到的，这一修改的过程实际上可能是4年后Legendre在1805年提出的最小二乘法的一个早期例子。
特别地，我们被建议用第一章中的方式根据 $\bold{r},\bold{r}''$ 计算 Ceres 的轨道参数。根据这些参数，我们可以得出一个理论上 Ceres 在 $\tau'$ 时刻的位置。这一位置对应于 $\lambda',\beta'$ 的新值，不妨记为 $\tilde{\lambda},\tilde{\beta}$。进一步，我们记 $\mathcal{L}= | \lambda' - \tilde{\lambda} |$，$\mathcal{B}= | \beta' - \tilde{\beta} |$，用来表示 $\tau'$ 时刻观测值与理论值之间的差别。然后，我们可以用 $\lambda,\lambda'-\mathcal{L},\lambda''$ 和 $\beta, \beta'-\mathcal{B}, \beta''$来重复Gauss的计算。有必要的话，可以重复迭代直到误差小到可以忽略。这是我们在计算中引入了一些看似粗放的近似的原因。

可以看到，这里跟我们今天理解的最小二乘法实际上也不是很像，而更像是根据迭代法求方程的数值解。不过高斯真实的想法还是要看他的原文可能才能更真实地理解。

## 结语

Piazzi在1801年的第一天发现了Ceres并在一些观测后因为太阳的原因跟丢了。高斯根据Piazzi的观测数据选择了三个点 $\tau,\tau',\tau''$。他首先根据已知的地球的运动和这几个观测数据估计了 $\delta'$，然后根据观测估计了 $\delta,\delta''$，并得到了 $\bold{r},\bold{r}''$。然后Gauss 假设 $\bold{r},\bold{r}''$ 是正确的情况下计算 Ceres 的轨道参数。 他验证了这一轨道是否和 $\tau'$ 处的观测数据相符，并重复迭代计算来调整估计。直到他得到一组与所有观测数据的误差都符合一定范围后，他计算了未来几个时刻 Ceres的位置，使得1802年的第一天，天文学家重新发现了 Ceres。
尽管这一思想可以被简单地描述，主要的步骤却十分复杂。我们发现想要理解 Gauss 是如何有如此特别的先见之明将通往胜利的必要的方程连接起来是十分困难的。作者也希望自己成功地认真解释了每一步并且给读者们带来了一些天文学秘密的insights，以及可能也领会到 the great mind of Carl Friedrich Gauss。

Laplace祝贺Gauss："The Duke of Brunswick has discovered more in his country than a planet: a super-terrestrial spirit in a human body".

在 200 Years of Least Squares Method 中，作者们提到是1801年的12月 Ceres 被重新发现，然后Gauss开始使用 LSM，但没有发表。

- Gauss Elimination 高斯为了证明最小二乘法中的线性方程组的可解性，在1809年首次清楚地描述了后来以他的名字命名的消元法。
- Gauss-Newton Method 在同一篇论文中，他解释了非线性最小平方误差问题如何在邻域内近似线性化得到一个近似解然后迭代修正。
- Laplace's central limit theorem. 1809年，Laplace发表了他的中心极限定理，表示任意概率函数，在求算数平均后，都趋于正态分布。
- 1823年，Gauss 发表了LSM上第二篇基础工作，Theoria combinationis observationum erronibus minimis obnoxiae 分两部分，其中包括最小二乘原则的修正，独立于概率函数，这在今天被叫作 Gauss-Markov Theorem。
- 1828年，Bessel从离散情形发现了最小二乘思想，正交性和 Euler-Fourier formulas 作三角级数估计的关系；后来Gram（1883）年拓展到连续情形，是Fourier series的 $L^2$ Hilbert space的基础。
- 1845年，Jacobi发表了他的用$R^2$中的连续转动求解 normal equations的方法。这些旋转引导了后来1950年代的Givens的三角化方法以及第一个稳定的特征值算法。
- 1900年，Karl Pearson把最小二乘和 $\chi^2$分布联系起来，带来了著名的$\chi^2-test$假设检验。
- 1958年，Householder的反射算法出现，替换了Givens中的旋转，带来了 QR 分解，以及由 Golub（1965），成为了今天最小二乘法的标准算法。

虽然看了这些介绍，我们发现我们还不是非常理解最小二乘法。我们发现信号处理中会讲的更加程序化一些，比如线性系统下最小二乘法怎么估计参数。这些还是放到维纳滤波器来写好了。感谢 GNU $\TeX macs$ 帮我输入公式省了大量时间。

At last, many thanks to $Daniel\text{ }Bed’at\check{s}$ .
这篇博客主要翻译自他的 [Bachelor Thesis-Gauss’ calculation of Ceres’ orbit](https://dspace.cuni.cz/bitstream/handle/20.500.11956/128180/130307045.pdf?sequence=1)
感谢他同意我翻译他的论文产生了这篇博客。如果有建议，我会改进这篇博客。

## 参考文献

> BEĎATŠ, Daniel. Gauss'calculation of Ceres' orbit. Bakalářská práce, vedoucí Tůma, Jiří. Praha: Univerzita Karlova, Matematicko-fyzikální fakulta, Katedra algebry.

此外，参考了其他的文献包括

> Jonathan Tennenbaum and Bruce Director, How Gauss Determined the Orbit of Ceres

但这篇写的很长，85页，但包括了很多立体几何，结论又不那么清晰。

> https://math.berkeley.edu/~mgu/MA221/Ceres_Presentation.pdf

这个课件比较完整的重复了计算过程，不过没有足够的记号的解释，看完上面的过程再看可能会很好理解。

下面这篇主要是介绍最小二乘法的历史，200年是从1805年算起。

> A. Abdulle and G. Wanner, Genève, Basel, 200 Years of Least Squares Method, 2005
