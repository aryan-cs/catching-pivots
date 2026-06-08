# PLAN.md

**Where Reasoning Models Commit to Deception**

A conference-level research plan. Companion to [`docs/proof.pdf`](docs/proof.pdf).

---

## 0. One-paragraph summary

Reasoning-trained language models routinely produce chains of thought (CoT) whose internal content is honest but whose final output is not, across sycophancy, sandbagging, insider-trading, and alignment-faking settings. Per-token residual-stream probes detect the unfaithfulness signal during generation; uniform activation-steering interventions can flip the output but impose a substantial fluency tax. The gap between these two threads is a question that the standing literature does not answer: is there a single discrete token in the CoT, the *pivot*, at which the model commits to the deceptive plan, and can single-token intervention at that token flip the trajectory at a strictly smaller distributional cost than uniform steering? This project formalises the pivot as a causal stopping time on a two-state latent model of CoT generation, proves a $\Theta(T)$ separation in the KL footprint of pivot-localized intervention versus uniform CAA, specifies an online CUSUM-based pivot detector with calibrated false-alarm control, and tests the resulting Pivot-Localized Intervention (PLI) against the four nearest baselines in the literature (CAA, CAST, DSAS, PIXEL) across four published deception scenarios and four open reasoning models. The deliverable is a paper and an accompanying code release.

## 1. Motivation, threat model, hypothesis

### 1.1 Motivation

The position taken in the recent Korbak et al. monitorability paper ([2507.11473](https://arxiv.org/abs/2507.11473)) is that current reasoning models route deceptive plans through human-readable CoT and that this channel is a privileged but fragile safety property. The Baker et al. OpenAI study ([2503.11926](https://arxiv.org/abs/2503.11926)) shows that putting optimisation pressure on the legible CoT causes the model to obfuscate while continuing to reward-hack at the same rate; black-box CoT monitoring is therefore fragile under training pressure. White-box residual-stream methods do not share this failure mode, since the underlying activations are not the surface that optimisation pressure shapes. The standing white-box work (Goldowsky-Dill et al. [2502.03407](https://arxiv.org/abs/2502.03407), Chrabaszcz et al. [2605.18549](https://arxiv.org/abs/2605.18549), Mirtaheri and Belkin [2603.17199](https://arxiv.org/abs/2603.17199)) shows that per-token probes work for detection and trajectory characterisation, but does not push to causal intervention. The standing intervention work (Panickssery et al. [2312.06681](https://arxiv.org/abs/2312.06681), Li et al. [2306.03341](https://arxiv.org/abs/2306.03341), Lee et al. [2409.05907](https://arxiv.org/abs/2409.05907), Ferrando et al. [2512.03661](https://arxiv.org/abs/2512.03661), Yu et al. [2510.10205](https://arxiv.org/abs/2510.10205)) acts at every post-prompt token, uses a smoothly probe-weighted scaling, or selects a single position by offline scan; none couples an online residual-stream probe with single-token intervention. That gap is the project.

### 1.2 Threat model

The adversary is a reasoning-trained LLM with the following capacities: (i) it generates an extended scratchpad CoT before producing an answer; (ii) the scratchpad may contain explicit deceptive reasoning, encoded reasoning, or honest reasoning followed by a deceptive answer; (iii) the model's activations are available to a white-box defender at inference time but the model's weights are frozen; (iv) the defender may intervene on the residual stream during generation but cannot retrain or fine-tune the model. The objective of the defender is to flip the final answer from deceptive to honest while preserving capabilities on unrelated tasks. We do not assume access to ground-truth honesty labels at inference time; the probe and detector are built from offline contrast data.

### 1.3 Hypotheses

The project tests four falsifiable hypotheses.

* **H1 (Pivot existence).** For prompts on which the model exhibits CoT-output unfaithfulness, the per-token residual-stream honesty probe trajectory exhibits a detectable change-point.
* **H2 (Causal sufficiency).** Single-token residual-stream intervention at the change-point flips the output from deceptive to honest at non-trivial rates; the same intervention at matched-probe-value non-pivot tokens does not.
* **H3 (Pareto dominance).** PLI dominates uniform CAA, smoothly-gated DSAS, prompt-conditional CAST, and offline-position-selected PIXEL on the flip-rate-versus-distributional-cost frontier.
* **H4 (Cross-architecture generality).** The pivot phenomenon and the PLI gain over baselines transfer across DeepSeek-R1-Distill-Qwen-{14B,32B}, QwQ-32B, and DeepSeek-R1-Distill-Llama-8B, with effect sizes consistent across families.

H1 is the prerequisite. H2 is the load-bearing causal claim. H3 is the headline efficiency claim. H4 is the generality claim a reviewer will demand.

## 2. Honest novelty delta

The framing must surface overlap with prior work in the first page. The contributions are narrowly the following.

1. **Formal model.** A two-state latent Markov model of CoT generation under deception, with the pivot as a stopping time on the latent honesty process. The model is small, falsifiable, and supports the pivot-order theorem in §5 of the proof.
2. **Causal pivot definition.** A definition of the pivot via single-token interventional sufficiency, distinct from the statistical first-passage rule that prior probe-trajectory work would yield by default. The latent, statistical, and causal pivots are formally related by the pivot-order theorem (Theorem 5.1 of the proof).
3. **KL footprint bound.** A proof that pivot-localized intervention's KL footprint on the CoT distribution is $O(\alpha^2 \sigma^2)$ where uniform CAA's footprint is $\Theta(T \alpha^2 \sigma^2)$. This is the formal version of the headline efficiency claim.
4. **Online CUSUM detector.** A sequential probability ratio test for the pivot with calibrated sequence-level false-alarm rate, suitable for use during decoding.
5. **Pareto-frontier evaluation against four nearest neighbours.** The PIXEL paper (offline static position), DSAS paper (online smooth gating), CAST paper (prompt-conditional), and standard uniform CAA together cover the three cells of the (position-sparse × probe-gated) design space; PLI occupies the fourth cell (position-sparse and online probe-gated). The experimental design holds layer, direction, and contrast set fixed to make the comparison clean.

What we explicitly do not claim:

* Per-token probing is not novel; see Goldowsky-Dill ([2502.03407](https://arxiv.org/abs/2502.03407)) and Chrabaszcz et al. ([2605.18549](https://arxiv.org/abs/2605.18549)).
* Sparse position-localized intervention is not novel; PIXEL ([2510.10205](https://arxiv.org/abs/2510.10205)) selects a single position via offline scan.
* Probe-derived per-token gating is not novel; DSAS ([2512.03661](https://arxiv.org/abs/2512.03661)) does it smoothly. The conjunction of the two, with an online sequential detector and a formal KL bound, is what is new.

The novelty is therefore narrow. The pitch should emphasise the conjunction and the empirical Pareto frontier, not isolated components.

## 3. Method

### 3.1 The two-state latent model

We treat CoT generation as a partially observed Markov process with two latent states, $H$ (honest) and $D$ (deceptive). $Z_1 = H$ almost surely; $D$ is absorbing in the simplest specification; the per-token residual-stream probe is a noisy Gaussian observation of $Z_t$; the next-token distribution is a mixture indexed by $Z_t$. The pivot is the stopping time $T^{\circ} = \inf\{t : Z_t = D\}$. See §4 of the proof.

The absorbing assumption is relaxed in the extended model (Remark 4.3 of the proof) by introducing a small return rate. Empirically, R1-style models exhibit honest-to-deceptive transitions with subsequent return signalled by *Wait,* or *Actually,* markers; the simplest analysis fixes the first sustained run as the pivot.

### 3.2 Pivot definitions

Three pivots are formally distinguished:

| Pivot | Definition | Observable | Use |
|---|---|---|---|
| Latent $T^{\circ}$ | First $t$ with $Z_t = D$ | No | Theoretical reference |
| Statistical $T^{\star}_{\tau}$ | First $t$ with $S_t > \tau$ | Yes (online) | First-pass online estimator |
| Causal $T^{\dagger}_{\Delta}$ | First $t$ with single-token edit causing $\Delta$-flip | Yes (counterfactual) | Ground truth for evaluation |

The pivot-order theorem (Theorem 5.1 of the proof) shows that under the two-state model and a monotone causal advantage assumption, $T^{\dagger}_{\Delta} \le T^{\star}_{\tau}$ with high probability; the statistical estimator therefore lags the causal pivot. This motivates a sharper online detector.

### 3.3 The CUSUM detector

We use the Page-Lorden CUSUM rule (Page 1954, Lorden 1971) with the two-Gaussian log-likelihood ratio per step. The detector is causal in the streaming sense, controls a calibrated sequence-level false-alarm rate via the threshold $b$, and dominates the static-threshold rule on the optimal detection delay/false-alarm tradeoff. Pseudocode in §7 of the proof, Algorithm 1.

### 3.4 The PLI update

Pivot-Localized Intervention (PLI) is the single-token residual-stream replacement

$$\hh_{\ell}(t) \leftarrow \hh_{\ell}(t) - c\, \thetab \quad \text{for } t = \widehat{T}$$

and the identity for all other $t$. The direction $\thetab$ is the mass-mean honesty vector at layer $\ell$. The KL bound (Theorem 6.2 of the proof) shows that PLI's distributional footprint is $O(c^2 \lambda_{\max})$ where uniform CAA's footprint is $\Theta(T c^2 \lambda_{\min})$.

## 4. Datasets

### 4.1 Contrast set for probe fitting

Following Goldowsky-Dill et al. ([2502.03407](https://arxiv.org/abs/2502.03407)), we generate paired prompts that vary only the system-prompt directive between truthful and deceptive variants on a fixed underlying question set. Concrete construction:

* Base question set: 4000 factual disagreement prompts adapted from Sharma et al. ([2310.13548](https://arxiv.org/abs/2310.13548)), 2000 Insider Trading scenario adaptations of Scheurer et al. ([2311.07590](https://arxiv.org/abs/2311.07590)), 2000 Sandbagging-style capability test prompts following van der Weij et al. ([2406.07358](https://arxiv.org/abs/2406.07358)).
* Each base prompt is paired into $(\pi^+, \pi^-)$ by appending an honest or deceptive system directive in the style of the published prompts of the cited works.
* Per-pair we sample one CoT per system directive and discard pairs whose final outputs disagree across honest/deceptive directives at the answer-content level less than 5% of the time (no deception is being elicited) or more than 95% of the time (the contrast is trivial).

Held-out splits: 70/15/15 train/val/test by prompt-base, with no overlap on the base question. We additionally hold out 20% of prompt bases for cross-scenario transfer evaluation.

### 4.2 Evaluation scenarios

Four scenarios, each producing approximately 1000 unfaithful traces per model, defined as traces in which (i) a CoT-acknowledgement classifier scores the scratchpad as containing the correct answer or honest reasoning, and (ii) the final output disagrees with that.

* **S1 Sycophancy under confident user disagreement.** Factual prompts where the user states the wrong answer with high confidence. Adapted from Sharma et al. ([2310.13548](https://arxiv.org/abs/2310.13548)) and Perez et al. ([2212.09251](https://arxiv.org/abs/2212.09251)).
* **S2 Sandbagging under capability-test framing.** Following van der Weij et al. ([2406.07358](https://arxiv.org/abs/2406.07358)) and the operational sandbagging variant of Benton et al.
* **S3 Alignment faking with disclosed retraining context.** Following Greenblatt et al. ([2412.14093](https://arxiv.org/abs/2412.14093)), simplified to the single-prompt scratchpad regime.
* **S4 Insider trading with explicit oversight.** Following Scheurer et al. ([2311.07590](https://arxiv.org/abs/2311.07590)).

### 4.3 Capability preservation set

A combined MMLU-subset (1000 questions) and GPQA-Diamond (198 questions) set, scored under each intervention coefficient to produce the budget-quality frontier called for by Pres et al. ([2410.17245](https://arxiv.org/abs/2410.17245)).

## 5. Models

| Tier | Model | Role |
|---|---|---|
| Primary | DeepSeek-R1-Distill-Qwen-32B | Headline model; published unfaithfulness rates |
| Primary | QwQ-32B | RL-trained comparison to distillation |
| Sibling | DeepSeek-R1-Distill-Qwen-14B | Cheap sweeps, scaling comparison |
| Cross-arch | DeepSeek-R1-Distill-Llama-8B | Architecture control |
| Negative | s1.1-32B | SFT-only control: isolates RL-induced from imitated deception |

A 70B distill is out of scope for the white-box analysis; a smaller-than-7B sweep is also out of scope as the published unfaithfulness signal is weak below 14B.

## 6. Baselines

The baselines correspond to the cells of the (sparse vs dense position) × (offline vs online gating) design space.

| Baseline | Position | Gate | Source |
|---|---|---|---|
| No intervention | dense | none | upper bound on deception |
| Uniform CAA | dense | none | Panickssery et al. ([2312.06681](https://arxiv.org/abs/2312.06681)) |
| ITI | dense | none, per-head | Li et al. ([2306.03341](https://arxiv.org/abs/2306.03341)) |
| CAST | dense | prompt-conditional | Lee et al. ([2409.05907](https://arxiv.org/abs/2409.05907)) |
| DSAS | dense | online smooth | Ferrando et al. ([2512.03661](https://arxiv.org/abs/2512.03661)) |
| PIXEL | sparse | offline scan | Yu et al. ([2510.10205](https://arxiv.org/abs/2510.10205)) |
| PLI (first crossing) | sparse | online threshold | This work |
| PLI (argmax) | sparse | online maximum | This work (ablation) |
| PLI (CUSUM) | sparse | online sequential | This work (headline) |

All baselines use the same layer $\ell$, direction $\thetab$, and contrast set. Coefficient $c$ is swept independently per baseline at five values per scenario.

## 7. Metrics

The headline metric is the *KL-conditional flip rate*: flip rate at a fixed KL footprint budget. Raw flip rate is reported as a diagnostic; it is inflatable by large $c$ at correspondingly large KL cost, as Pres et al. ([2410.17245](https://arxiv.org/abs/2410.17245)) document.

| Metric | Definition | Direction |
|---|---|---|
| Flip rate | Fraction of trials where final output transitions deceptive to honest | Higher is better |
| KL footprint | Chain-rule estimate of $\KL(p_{\mathrm{int}} \| p_{\mathrm{base}})$ over the CoT | Lower is better |
| Perplexity | Token-level perplexity of the intervened CoT under base | Lower is better |
| MMLU/GPQA | Accuracy under intervention | Preserve baseline |
| Pivot delay | $\E[\widehat{T} - T^{\dagger}_{\Delta_0}]$, evaluated on a counterfactually-pivoted subset | Lower in magnitude is better |
| Sequence FA | False-alarm rate of CUSUM on honest CoT | Calibrated to 1% or 5% |
| Cross-scenario AUROC | Probe trained on S1+S2, tested on S3+S4 and vice versa | Higher is better |

Two independent rubric-based LLM judges (Gemini 2.5 Pro and Claude Sonnet 4.7) score flip events on disjoint prompts; inter-judge agreement is reported.

## 8. Ablations

| Tag | Purpose | Description |
|---|---|---|
| A | Pivot estimator | CUSUM vs first-crossing vs argmax vs random-token-at-matched-probe vs all-token |
| B | Layer sweep | $\ell$ at five points around the validation-AUROC argmax, connecting to the LAP profile of Billa ([2604.15557](https://arxiv.org/abs/2604.15557)) |
| C | Coefficient | $c \in \{0.5, 1, 2, 4, 8\} \cdot \widehat{\sigma}_{\ell}$ |
| D | Probe family | mass-mean vs logistic regression vs CCS (Burns et al. [2212.03827](https://arxiv.org/abs/2212.03827)) vs Truth-is-Universal two-dimensional (Bürger et al. [2407.12831](https://arxiv.org/abs/2407.12831)) |
| E | Placebo | Edit with opposite-direction vector at the CUSUM pivot. Expected null effect. |
| F | Steganography stress test | Paraphrase the CoT in flight following Roger and Greenblatt ([2310.18512](https://arxiv.org/abs/2310.18512)); check whether pivot persists in residual stream when legible scratchpad has been obfuscated |
| G | Reversible model | Replace absorbing-state two-state model with reversible model; redo pivot order analysis empirically |

Ablation A is the load-bearing one: it isolates the contribution of the sequential detector over the underlying probe. Ablation F is the most safety-relevant one: it tests the Baker et al. obfuscation failure mode directly.

## 9. Statistical analysis

Each design cell (model × scenario × baseline × coefficient) is replicated with at least 256 prompts and 16 seeds per prompt. We report bootstrap 95% confidence intervals at the prompt level (cluster bootstrap) and at the seed level. Pareto-frontier plots use the dominated-set criterion: a baseline is Pareto-dominated if at every flip-rate target it has a higher KL footprint than some other baseline. We expect PLI (CUSUM) to Pareto-dominate at flip-rate targets above $\eta^{\ast} \approx 0.3$ and to be dominated by uniform CAA at very high flip-rate targets, since uniform CAA's saturating coefficient buys more flips at the cost of capability degradation.

Multiple-comparisons control: we adopt Holm-Bonferroni across the table of pairwise PLI vs baseline comparisons at the headline metric. Family-wise error rate target is 5%.

## 10. Principal risks

The principal risks to the empirical programme are: (i) the unfaithful-trace yield in S3 is too low on open models (mitigation: substitute Apollo's covert-email-reranking scenario); (ii) the CUSUM detector has insufficient detection power at the published unfaithfulness rates on 32B models (mitigation: relax to a Bayesian online change-point detector with a stronger prior, see Adams and MacKay style); (iii) the steganography ablation requires modifying the decoding pipeline non-trivially (mitigation: implement as a post-hoc reroll comparison rather than in-line paraphrasing).

## 11. Engineering plan

The code is structured around a streaming-hook abstraction over a HuggingFace transformers forward pass; vLLM and SGLang are used for the high-throughput baseline scoring sweeps, with a fallback to plain transformers for the activation-edit experiments where the optimised servers do not yet expose a clean hook.

```
catching-pivots/
├── PLAN.md                     ← this file
├── README.md                   ← repo landing page
├── docs/
│   ├── proof.tex               ← formal theory
│   └── proof.pdf               ← compiled
├── pivots/                     ← Python package
│   ├── probes/                 ← mass-mean, logistic, CCS, Truth-is-Universal
│   ├── detectors/              ← first-crossing, argmax, CUSUM
│   ├── interventions/          ← PLI, CAA, ITI, CAST, DSAS, PIXEL adapters
│   ├── scenarios/              ← S1-S4 prompt generators and filtering
│   ├── eval/                   ← flip rate, KL footprint, capability, perplexity
│   └── runtime/                ← residual-stream hook harness
├── experiments/                ← scripted sweeps; one file per ablation
├── data/                       ← contrast sets, scenario splits, capability sets
└── tests/
```

The intervention API has the shape

```python
with model.streaming_hook(layer=24) as h:
    detector = CUSUM(direction=theta, mu_h=mu_h, mu_d=mu_d, sigma=sigma, b=b)
    for token, hidden in model.generate_stream(prompt, max_new_tokens=4096):
        if not detector.fired and detector.update(hidden):
            h.edit(hidden, -c * theta)
            detector.fire()
        yield token
```

This puts the streaming detector and the single-token edit on the same side of the abstraction barrier as model generation, which is necessary for the intervention to act on the *forward-pass* activations of the token whose probe score triggered firing rather than on the next token.

## 12. What success and what failure look like

**Success** is a Pareto plot in which PLI (CUSUM) sits below and to the right of the CAA, ITI, CAST, DSAS, and PIXEL curves on at least three of the four scenarios and at least three of the four primary models, with capability preservation at flip-rate targets within 2 absolute points of the no-intervention baseline. The accompanying theoretical claim (the KL bound of Theorem 6.2) is supported by the measured KL footprints scaling sublinearly in CoT length while CAA's scales linearly.

**Partial success** is the same Pareto plot on a single model or single scenario, with the rest showing PLI matching the best baseline. The contribution narrows to "in some regimes, this works." Honest reporting is required.

**Failure** has three possible shapes. (i) The CUSUM detector cannot achieve a usable false-alarm-versus-delay tradeoff on the open models, in which case the detection layer of the work fails and the contribution reduces to an offline characterisation of the causal pivot via counterfactual edits. (ii) The pivot is not causally sufficient at intervention magnitudes that preserve capabilities; in this case the H2 hypothesis is falsified and the paper is honest about it, reporting under what conditions single-token intervention is and is not sufficient. (iii) The pivot phenomenon is present but PLI does not Pareto-dominate DSAS or PIXEL; the paper then becomes a refined characterisation of when each baseline is preferred and an ablation-driven analysis of the design tradeoffs, rather than a method paper. Each of these is a publishable outcome at a workshop, though only the first is a strong main-venue paper.

## 13. Open problems and future work

* **Multi-pivot models.** A CoT with multiple deceptive segments separated by honest reasoning is not handled by the absorbing-state model. The Bayesian online changepoint formulation of Adams-MacKay would generalise; the cost is a heavier per-token detector and a more elaborate ablation story.
* **Subspace pivots.** The mass-mean direction may not be the right object if honesty is multi-dimensional, as Bürger et al. ([2407.12831](https://arxiv.org/abs/2407.12831)) suggest. A subspace-projection version of PLI is natural.
* **Adversarial robustness.** Whether PLI's defence survives RL training pressure against the white-box detector itself, analogous to Baker et al.'s monitor-evasion result on the black-box side, is the key generalisation question and the natural follow-up.
* **Reasoning model weights ablation.** Whether the pivot is associated with a specific subcircuit (e.g., the attention heads that read scratchpad content) or is distributed; activation patching across $(\ell, t)$ pairs centred on the pivot is the standard tool.

## 14. Reproducibility commitments

* Code released under MIT or Apache-2.0 license; license decided at submission.
* All scenario prompts and probe contrast pairs released as a dataset.
* Per-cell seed and prompt manifests released as JSON-Lines artefacts.
* Pre-registered analysis plan for the headline Pareto comparison, deposited at the project repository commit hash at submission.
* No claim of state-of-the-art deception detection or steering is made; the paper is framed as an intervention-efficiency result with a specific causal characterisation, not a benchmark paper.

## 15. Related work in one paragraph

The closest neighbours by problem statement are Goldowsky-Dill et al. ([2502.03407](https://arxiv.org/abs/2502.03407)), Chrabaszcz et al. ([2605.18549](https://arxiv.org/abs/2605.18549)), and Mirtaheri and Belkin ([2603.17199](https://arxiv.org/abs/2603.17199)) on probing; Panickssery et al. ([2312.06681](https://arxiv.org/abs/2312.06681)), Li et al. ([2306.03341](https://arxiv.org/abs/2306.03341)), Lee et al. ([2409.05907](https://arxiv.org/abs/2409.05907)), Ferrando et al. ([2512.03661](https://arxiv.org/abs/2512.03661)), and Yu et al. ([2510.10205](https://arxiv.org/abs/2510.10205)) on intervention; and Baker et al. ([2503.11926](https://arxiv.org/abs/2503.11926)), Chen et al. ([2505.05410](https://arxiv.org/abs/2505.05410)), and Korbak et al. ([2507.11473](https://arxiv.org/abs/2507.11473)) on the broader monitorability frame. Foundational threads include Burns et al. ([2212.03827](https://arxiv.org/abs/2212.03827)) on contrast-consistent search, Zou et al. ([2310.01405](https://arxiv.org/abs/2310.01405)) on representation engineering and LAT, Marks and Tegmark ([2310.06824](https://arxiv.org/abs/2310.06824)) on the geometry of truth, MacDiarmid et al. (Anthropic) on simple-probe defection detection, Meinke et al. ([2412.04984](https://arxiv.org/abs/2412.04984)) and Greenblatt et al. ([2412.14093](https://arxiv.org/abs/2412.14093)) on scheming and alignment faking, and Lanham et al. ([2307.13702](https://arxiv.org/abs/2307.13702)) and Turpin et al. ([2305.04388](https://arxiv.org/abs/2305.04388)) on CoT faithfulness measurement. Pres et al. ([2410.17245](https://arxiv.org/abs/2410.17245)) and Li et al. ([2603.24543](https://arxiv.org/abs/2603.24543)) set the evaluation protocol for behaviour steering we adopt. The proof document (`docs/proof.pdf`) discusses the placement of the contribution relative to each of these in §2.

## 16. Authorship and acknowledgments

Lead: Aryan Gupta (`aryan.cs.app@gmail.com`). Advisors and collaborators to be added during the empirical phase. Acknowledgments will recognise Apollo Research, Anthropic Alignment Science, and OpenAI's monitorability work for setting the empirical baselines this project compares against, without implying endorsement.
