---
layout: post
title: Three example of coupling techniques, Cédric Villani
date: 2026-06-22 00:25:00
description: part of chapter 2 of Optimal Transport, Old and New
tags: OT
categories: math
---

<body>
    <table class="title-block" style="margin-bottom: 2em">
      <tr>
        <td><table class="title-block" style="margin-top: 0.5em; margin-bottom: 0.5em">
          <tr>
            <td><font style="font-size: 168.2%"><strong>Optimal Transport, Old and
            New</strong></font></td>
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
                        <class style="font-variant: small-caps">by C&eacute;dric Villani</class>
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
    <h2 id="auto-1">2<span style="margin-left: 1em"></span>Three examples of coupling techniques<span
    style="margin-left: 1em"></span></h2>
    <p>
      <div style="display: inline">
        <a id="auto-2"></a>
      </div>
      <h5>Convergence of the Langevin process<span style="margin-left: 1em"></span></h5>
    </p>
    <p>
      Consider a particle subject to the force induced by a potential \(V \in
      C^1 (\mathbb{R}^n)\), a friction and a random white noise agitation. If
      \(X_t\) stands for the position of the particle at time \(t\), \(m\) for
      its mass, \(\lambda\) for the friction coefficient, \(k\) for the
      Boltzmann constant and \(T\) for the temperature of the heat bath, the
      Newton's equation of motion can be written
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle m \frac{d^2 X_t}{d t^2} = - \nabla V
        (X_t) - \lambda m \frac{d X_t}{d t} +
\sqrt{k T} \frac{d B_t}{d
        t},\)</td>
        <td align="right">(2.1)</td>
      </tr>
    </table>
    <p>
      where \((B_t)_{t \geq 0}\) is a standard Brownian motion. This is a
      second order (stochastic) differential equaiton, so it should come with
      initial conditions for both the position \(X\) and the velocity
      \(\dot{X}\).
    </p>
    <p>
      Now consider a large cloud of particles evolving independently,
      according to (2.1); the question is whether the distribution of
      particles will converge to a definite limit as \(t \rightarrow \infty\).
      In other words: Consider the stochastic differential equation (2.1)
      starting from some intial distribution \(\mu_0 (d x d v)
      =\operatorname{law} (X_0, \dot{X_0})\); is it true that law(\(X_t\)), or
      law(\(X_t, \dot{X_t}\)) will converge to some given limit law as \(t
      \rightarrow \infty\)?
    </p>
    <p>
      Obviously, to solve this problem, one has to make some assumptions on
      the potential \(V\), which should prevent the particles from all
      escaping at infinity; for instance, we can make the very strong
      assumption that \(V\) is uniformly convex, i.e. there exists \(K > 0\)
      such that the Hessian \(\nabla^2 V\) satisfies \(\nabla^2 V \geq K
      I_n\). Some assumption on the initial distribution might also be needed:
      for instance, it is natural to assume that the Hamiltonian has finite
      expectation at initial time:
    </p>
    <center>
      \(\displaystyle \mathbb{E} \left( V (X_0) + \frac{| \dot{X} |^2}{2}
      \right) < + \infty\)
    </center>
    <p>
      Under these assumptions, it is true that there is exponential
      convergence to equilibrium, at least if \(V\) does not grow too wildly
      at infinity (for instance if the Hessian of \(V\) is also bounded
      above). However, I do not know of any simple method to prove this.
    </p>
    <p>
      On the other hand, consider the limit where the friction coefficient is
      quite strong, and the motion of the particle is so slow that the
      acceleration term may be neglected in front of the others: then, up to
      resetting units, equation (2.1) becomes
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \frac{d X_t}{d t} = - \nabla V (X_t)
        + \sqrt{2} \frac{d B_t}{d t},\)</td>
        <td align="right">(2.2)</td>
      </tr>
    </table>
    <p>
      which is often called a Langevin process. Now, to study the convergence
      of equilibrium for (2.2) there is an extremely simple solution by
      coupling. Consider another random position \((Y_t)_{t \geq 0}\) obeying
      the same equation as (2.2):
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \frac{d Y_t}{d t} = - \nabla V (Y_t)
        + \sqrt{2} \frac{d B_t}{d t} .\)</td>
        <td align="right">(2.3)</td>
      </tr>
    </table>
    <p>
      where the random realization of the Brownian motion is the same as in
      (2.2) (this is the coupling). The initial positions \(X_0\) and \(Y_0\)
      may be coupled in an arbitrary way, but it is possible to assume that
      they are independent. In any case, since they are driven by the same
      Brownian motion, \(X_t\) and \(Y_t\) will be correlated for \(t > 0\).
    </p>
    <p>
      Since \(B_t\) is not differentiable as a function of time, neighther
      \(X_t\) nor \(Y_t\) is differentiable (equations (2.2) and (2.3) hold
      only in the sense of solutions of stochastic differential equations);
      but it is easily checked that \(\alpha_t := X_t - Y_t\) is a
      continuously differentiable function of time, and
    </p>
    <center>
      \(\displaystyle \frac{d \alpha_t}{d t} = - (\nabla V (X_t) - \nabla V
      (Y_t)),\)
    </center>
    <center>
      \(\displaystyle \frac{d}{d t} \frac{| \alpha_t |^2}{2} = - \langle
      \nabla V (X_t) - \nabla V
(Y_t), X_t - Y_t \rangle \leq - K | X_t - Y_t
      |^2 = - K | \alpha_t |^2 .\)
    </center>
    <p>
      It follows by Gronwalls lemma that
    </p>
    <center>
      \(\displaystyle | \alpha_t |^2 \leq e^{- 2 K t} | \alpha_0 |^2 .\)
    </center>
    <p>
      Assume for simplicity that \(\mathbb{E} | X_0 |^2\) and \(\mathbb{E} |
      Y_0 |^2\) are finite. Then
    </p>
    <table width="100%">
      <tr>
        <td width="100%" align="center">\(\displaystyle \mathbb{E} | X_t - Y_t |^2 \leq e^{-
        2 K t} \mathbb{E} | X_0 - Y_0 |^2 \leq 2
(\mathbb{E} | X_0 |^2
        +\mathbb{E} | Y_0 |^2) e^{- 2 K t} .\)</td>
        <td align="right">(2.4)</td>
      </tr>
    </table>
    <p>
      In particular, \(X_t - Y_t\) converges to 0 almost surely, and this is
      independent of the distribution of \(Y_0\).
    </p>
    <p>
      This in itself would be essentially sufficient to guarantee the
      existence of a stationary distribution: but in any case, it is easy to
      check, by applying the diffusion formula, that
    </p>
    <center>
      \(\displaystyle \nu (d y) = \frac{e^{- V (y)} d y}{Z}\)
    </center>
    <p>
      (where \(Z = \int e^{- V}\) is a normalization constant) is staionary:
      If \(\operatorname{law} (Y_0) = \nu\), then also \(\operatorname{law}
      (Y_t) = \nu\). Then (2.4) easily implies that \(\mu_t :=
      \operatorname{law} (X_t) \) converges weakly to \(\nu\); in addition,
      the convergence is exponentially fast.
    </p>
    <p>
      <div style="display: inline">
        <a id="auto-3"></a>
      </div>
      <h5>Euclidean isoperimetry<span style="margin-left: 1em"></span></h5>
      <div style="display: inline">
        skipped.
      </div>
    </p>
    <p>
      <div style="display: inline">
        <a id="auto-4"></a>
      </div>
      <h5>Caffarelli's log-concave perturbation theorem<span style="margin-left: 1em"></span></h5>
      <div style="display: inline">
        skipped
      </div>
    </p>
  </body>
