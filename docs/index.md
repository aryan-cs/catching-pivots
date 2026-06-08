---
title: Where Reasoning Models Commit to Deception
---

# Where Reasoning Models Commit to Deception

- [Read the proof (PDF)](proof.pdf)
- [Read the plan](https://github.com/aryan-cs/catching-pivots/blob/master/PLAN.md)
- [Source on GitHub](https://github.com/aryan-cs/catching-pivots)

Reasoning-trained language models routinely produce chains of thought whose intermediate content is honest but whose final answer is not. This project formalises the *pivot*: the discrete token at which the model commits to the deceptive plan. The theory proves a $\Theta(T)$ separation between the Kullback--Leibler footprint of pivot-localized intervention and uniform Contrastive Activation Addition, and gives an online CUSUM detector with calibrated false-alarm control. The empirical programme tests the resulting Pareto frontier against the four nearest published baselines (CAA, CAST, DSAS, PIXEL) across sycophancy, sandbagging, alignment-faking, and insider-trading scenarios on open reasoning models.

Author: Aryan Gupta (`aryan.cs.app@gmail.com`).

Copyright © 2026 Aryan Gupta. Licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
