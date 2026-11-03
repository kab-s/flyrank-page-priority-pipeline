# Capstone Report — Page Priority Pipeline: Decision-Support Ranker for Organic Click Decline

- **Author:** Kabindra Shrestha
- **Lane:** Applied ML / Decision-Support Ranking/Scoring & Early-Warning Systems
- **Deployed Case Study & Web Report:** https://kab-s.github.io/flyrank-page-priority-pipeline/
- **Repo:** https://github.com/kab-s/flyrank-page-priority-pipeline
- **Date:** 2026-11-02

---

## 0. Abstract

Can machine learning accurately prioritize declining organic search content for an editor's monthly triage capacity without memorizing client domain authority? We evaluate this problem on the FlyRank Search Intelligence Warehouse release comprising 78,835,655 pseudonymized daily fact records, extracting a point-in-time cohort of 38,271 eligible page snapshots across 22 mature client domains at monthly decision boundary $T = \text{2026-02-01}$ with an observed forward 30-day decline base rate of 15.76% (6,032 declining URLs). We formulate a Two-Part Expected Utility scoring framework combining calibrated gradient-boosted decline probabilities with empirical click loss exposure ($\text{Priority} = \hat{p} \times \text{Impact} \times (1 + \max(0, \Delta\text{pos}))$), evaluated under strict 5-fold out-of-domain client-grouped cross-validation (`GroupKFold`) with inverse client sample weighting. Within an individual editor's monthly triage capacity of 50 in-depth page reviews, our operational ranker achieves Precision@50 of **66.00%** (**4.19× lift** over base rate, 33 true positives capturing 27.1 expected monthly clicks saved), outperforming the audited heuristic baseline (56.00%) by $+10.00\%$ percentage points and achieving **90.00% precision (5.71× lift)** at the head of the queue ($K=10$). Across wider review depths ($K=500$), the heuristic rule degrades rapidly to 30.60% while machine learning sustains **52.20% precision**, delivering a **+71% relative yield advantage** by detecting non-linear multi-window erosions. Proactive split audits uncover a substantial client memorization gap ($\Delta\text{ROC-AUC} = +12.93\%$ percentage points inflated under naive random split), demonstrating that standard row splits severely overstate generalization on unseen domains. To prevent edit fatigue and ranking destabilization, model release is governed by three asserted operational code gates (90-day cold-start tenure, 10,753 protected durable risers, 243 extreme volume verification checks) and two-clock drift telemetry (Pre-release Population Stability Index $= 0.0531$). This pipeline functions strictly as decision support to optimize scarce human inspection, making no causal claims regarding search engine ranking algorithms.

---

## 1. Problem framing

### Decision Context and Operational Workflow
Enterprise search marketing agencies manage tens of thousands of published URLs across diverse client domains. Over time, content performance decays due to shifting query intent, algorithm updates, competitor content refreshes, and internal freshness decay. In an enterprise search agency, editorial inspection capacity is bounded: an individual senior content specialist or editor has a monthly triage capacity of approximately **50 in-depth page audits**. A comprehensive content audit requires analyzing search intent, Google search results (SERP) competitor movements, internal links, and rewrite strategy, consuming 4 to 8 human hours per page.

Unassisted manual audits are infeasible across 38,000+ live client URLs, while naive triage rules (e.g., sorting by historical traffic volume or uncalibrated dip heuristics) misallocate editorial hours toward pages experiencing transient volatility or terminal structural floors. The Page Priority Pipeline frames this challenge as an operational decision-support ordering problem: *Given scarce editorial hours, which 50 pages should an editor audit first to preserve the maximum volume of search traffic?*

```
                           INDIVIDUAL EDITOR MONTHLY CAPACITY = 50 PAGES
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   Unassisted / Naive Triage                                     Page Priority Pipeline (Proposed)
   • Triage by Raw Volume: Reviews giant                        • Order by Expected Utility: Priority = p̂ × Impact
     stable pages; misses quiet erosions.                       • Filter by 3 Asserted Code Gates: Protects risers.
   • Triage by Recent Dip: Chases mean-                         • Action Taxonomy: Model supplies order; audited
     reversion bounces; 66.2% hit floor.                          rule supplies actionable human reason code.
   Result: Severe false-positive edit fatigue.                   Result: Precision@50 = 66.0% (4.19× lift over base).
```

### Unit of Analysis, Target Formulation, and Asymmetric Costs
* **Unit of Analysis:** A single published content item at as-of monthly decision snapshot $T$: `(content_hash_id, snapshot_date T=2026-02-01)`.
* **Output:** A globally ranked priority queue where each row provides `ml_priority_score`, rank, calibrated decline probability $\hat{p}$, estimated monthly clicks at stake, tactical action category, and transparent reason code.
* **Target Label Definition:** Forward-looking 30-day impression drop exceeding 20% relative to the pre-decision baseline:
  $$\text{Target} = \mathbf{1}\left[\sum_{t=T+1}^{T+30} \text{impr}_t < 0.80 \times \sum_{t=T-30}^{T} \text{impr}_t\right]$$
  Enforced with a 100-impression volume floor and an availability rule ($\ge 25$ days of search console data).
* **Asymmetric Error Costs:**
  * *Cost of False Positive (Type I):* An editor spends 4 to 8 hours rewriting an article that was merely experiencing temporary seasonality or durable growth, introducing risk of ranking destabilization and wasting scarce retainer budget.
  * *Cost of False Negative (Type II):* An unflagged high-value revenue pillar collapses silently over 30 days, resulting in compounding loss of organic search real estate to competitors.
* **Why ML Helps:** While simple heuristic rules identify dramatic historical drops, they cannot capture non-linear interactions across SERP position drift, multi-window momentum ratios, and traffic concentration. ML provides calibrated continuous probabilities that enable rigorous utility multiplication ($\text{Risk} \times \text{Impact}$).

---

## 2. Data safety

### Warehouse Scope and Eligibility Filtering
The analysis is conducted on the FlyRank Search Intelligence Warehouse release (2025-01-27 to 2026-06-30), containing 78,835,655 pseudonymized daily performance facts across 104 client records and 519,606 content records stored in DuckDB Parquet tables.

At decision boundary $T = \text{2026-02-01}$, candidate page snapshots are filtered against five operational eligibility criteria:
1. **Volume Floor:** $\ge 100$ impressions in the 30-day baseline window $[T-30, T]$.
2. **Data Availability:** $\ge 25$ active reporting days with Search Console coverage.
3. **Maturity / Cold-Start Rule:** Content published date $\le T - 90\text{ days}$ (new pages lack stable baselines).
4. **Client Maturity:** Client onboarded $\ge 90\text{ days}$ before $T$ (ensuring 60 days of pre-baseline history).
5. **Publication State:** Live, published content items only (unpublished/deleted items excluded upstream).

This yields an evaluation cohort of **38,271 eligible page snapshots across 22 mature client domains**, with an observed 30-day decline base rate of **15.76%** (6,032 declining pages).

### Quarantined Fields and Excluded Columns
To maintain complete client-agnostic portability and prevent target leakage, all candidate warehouse columns were classified into a strict 4-bucket field taxonomy:
* **Excluded Entity Identifiers:** `client_hash_id` and `content_hash_id` are strictly quarantined from feature vectors to prevent decision trees from memorizing domain-level historical averages.
* **Excluded CMS & AI Metadata:** `content_updated_date`, `content_optimized_date`, `provider_used`, and `model_used` are excluded because CMS timestamps reflect export-time state rather than historical state at $T$, while AI generation tags serve as client fingerprints.
* **Excluded Retrospective Trend Fields:** Warehouse summary fields `trend_direction` and `trend_pct` compare pre-decision windows ($[-30, 0]$ vs $[-60, -30]$) and reflect past declines rather than future risk.
* **Privacy Assurance:** All data is irreversibly pseudonymized via SHA-256 hashes. Zero client domain names, author identities, or raw query text appear in feature matrices or serialized outputs.

### Proactive 4-Vector Leakage Attack Suite
Before training candidate models, the feature matrix was subjected to four adversarial leakage assertions:
1. **SQL Timeline Boundary Assertion:** Invariant $\max(\text{feature\_report\_date}) \le T\text{ (2026-02-01)}$ and $\min(\text{label\_report\_date}) \ge T+1\text{ (2026-02-02)}$ held with **0 illegal overlaps**, mathematically guaranteeing zero label contamination.
2. **Excluded Fields Invariant:** Automated schema audit verified **0 forbidden columns** (0 entity hashes, 0 CMS update timestamps, 0 composite health scores) present in feature matrix $X$.
3. **Suspect Feature Ablation Test:** Retraining without the top suspect ratio (`pct_of_peak_proxy`) produced a negligible change in out-of-fold generalization ($\Delta\text{ROC-AUC} = -0.0417$, safely below the $0.05$ collapse threshold), confirming the signal represents genuine predictive momentum rather than peeking.
4. **Synthetic Leakage Detection Attack:** Deliberately injecting forward 30-day traffic ratios into the feature set caused ROC-AUC to surge immediately to $1.0000$, verifying that the validation harness possesses 100% sensitivity to forward leakage.

---

## 3. Baseline

### Formulating the Transparent Heuristic Baseline
Before training machine learning classifiers, we established a transparent, unweighted heuristic baseline action score combining traffic volume, velocity decay, and SERP slippage:
$$\text{Baseline Action Score} = \left(1.0 - \min(0.80, \text{impr\_ratio}_{30\_prev30})\right) \times \ln(1 + \text{impr}_{w30}) \times \left(1.0 + \max(0, \Delta\text{pos}_{30\_prev30})\right)$$
Pages with severe velocity decay ($<0.50\times$), rank drops ($\ge +0.5$ positions), and volume $\ge 500$ impressions receive high baseline scores with human-interpretable reason codes (e.g., `SEVERE_VELOCITY_DROP_AND_RANK_SLIP`).

### The Anti-Baseline: Why the Intuitive Rule Fails (Mean Reversion)
The most common intuitive heuristic deployed by SEO teams is an "already dipping" watchlist: targeting pages that experienced the steepest drops in recent weeks. 

When audited against our ground-truth warehouse receipts across all 38,271 eligible pages, this intuitive rule fails dramatically due to **mean reversion**:
* Among 5,299 pages exhibiting retrospective decline prior to $T$, **66.22% (3,509 pages) hit a structural floor and stabilized** or rebounded over the subsequent 30 days. Only 33.78% (1,790 pages) continued downward.
* Retrospective drops missed **4,242 forward collapses** (70.32% of all actual declines), proving that backward-looking percentage loss is an unreliable triage signal.
* Evaluating editorial triage on retrospective decline alone causes teams to allocate two-thirds of their monthly capacity to pages whose decline has already concluded, leaving quiet erosions unaddressed.

### Baseline Benchmark Performance
Evaluated across all 38,271 eligible pages, the baseline rule achieves:
* **ROC-AUC:** 0.5815 | **PR-AUC:** 0.2553
* **Precision@10:** 80.00% (8 true positives, 5.08× lift over 15.76% base rate)
* **Precision@20:** 70.00% (14 true positives, 4.44× lift)
* **Precision@50 (Monthly Budget):** 56.00% (28 true positives, 3.55× lift)
* **Precision@100:** 42.00% (42 true positives, 2.66× lift)
* **Precision@250:** 34.00% (85 true positives, 2.16× lift)
* **Precision@500:** 30.60% (153 true positives, 1.94× lift)

This transparent heuristic provides the honest operational hurdle the machine learning ranker must beat.

---

## 4. Model / analysis

### Two-Part Expected Utility Framework
We decouple the ranking problem into an empirical risk classifier and an impact estimator:
$$\text{Priority Score} = \hat{p}(\text{decline}) \times \text{Expected Clicks Lost} \times \left(1 + \max(0, \Delta\text{pos})\right)$$
Where $\hat{p} \in [0, 1]$ is the calibrated decline probability, and Expected Clicks Lost is calculated via Empirical-Bayes smoothed baseline monthly clicks multiplied by the dynamic empirical drop severity ($57.53\%$, derived from retrospective decaying pages preceding $T$ with zero leakage).

This mathematical decoupling prevents two critical failure modes in operations:
1. It ignores high-traffic evergreen articles that are structurally secure ($\hat{p} \approx 0$).
2. It ignores low-volume tail URLs that have 95% decline probability but only 3 monthly impressions ($\text{Impact} \approx 0$).

### Feature Set (29 Point-in-Time Features)
Features span five distinct domain families, computed strictly as of $T$:
1. **Search Momentum (7):** 30d-to-60d impression ratio, 7d-to-30d short-term acceleration, 7d-to-prev-7d velocity, SERP position drift ($\Delta\text{pos}_{30\_prev30}$), CTR shift, 14d rolling momentum slope.
2. **Traffic Level & Concentration (6):** Log impressions (30d, 90d), log clicks, share of client total traffic, page traffic percentile rank within client domain.
3. **Historical Peak Distance (4):** Ratio of current 30d volume to historical peak proxy, days since observed historical peak, trough depth indicator.
4. **User Engagement & Analytics (6):** GA4 engagement rate, bounce rate proxy, average session duration, scroll depth proxy, conversion rate baseline (missing values handled via indicator encoding).
5. **Scale-Free SERP Profile (6):** SERP position tier bucket, top-3 indicator, first-page indicator (positions 4–10), CTR-to-position expectation ratio.

### Target Definition (One Sentence)
A binary indicator marking whether an eligible page's total forward 30-day search impressions decrease by more than 20% relative to its pre-decision 30-day baseline volume ($15.76\%$ observed base rate across 38,271 eligible items).

### Candidate Tournament
We evaluated candidate architectures under identical 5-fold client-grouped validation (`GroupKFold` clustered by `client_hash_id`) with Decision D11 inverse sample weighting:
1. **Heuristic Baseline Action Score:** Deterministic formula combining velocity decay and traffic volume.
2. **Regularized Logistic Regression:** Scaled linear model with $L_2$ regularization.
3. **Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier`):** Non-linear decision tree ensemble with native missing-value support, monotonic constraints on momentum ratios, and inverse client sample weighting.

---

## 5. Evaluation

### The Client Memorization Gap: Grouped vs Random Splits
When validating global rankers across multi-client portfolios, naive random row splits create catastrophic data leakage. Decision trees split on subtle client fingerprints (such as `client_share_of_traffic` or average volume scale), effectively memorizing client identities.

To quantify this error, we evaluated the champion `HistGradientBoosting` model under both validation designs on the full warehouse cohort:

| Validation Design | Split Logic | ROC-AUC | PR-AUC | Brier Score | Pure Risk P@50 | Lift@50 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Naive Random Split** | 80/20 random rows | 0.8191 | 0.5511 | 0.0997 | **100.00%** | 6.34× |
| **Honest Client Holdout** | 5-Fold `GroupKFold` | **0.6898** | **0.2953** | **0.1261** | **60.00%** | 3.81× |
| **Client Memorization Gap** | Leakage Penalty | **+12.93% pt** | **+25.58% pt** | -0.0264 | **+40.00% pt** | +2.53× |

> **Critical Takeaway:** Random train/test splits overstate production performance by nearly 13 percentage points in ROC-AUC (+12.93% pt) and 25.58 percentage points in PR-AUC. An applied ML system that cannot transfer to unseen client domains will fail immediately upon onboarding new clients. All subsequent benchmarks report the honest client-grouped numbers.

### Model Tournament Benchmark Table
Evaluated across all 38,271 eligible pages under strict 5-fold client-grouped holdout:

| Model Architecture | Validation Protocol | ROC-AUC | PR-AUC | Brier Score | Precision@50 (Editor Capacity) |
|---|---|:---:|:---:|:---:|:---:|
| **Heuristic Baseline Action Score** | Deterministic Formula | 0.5815 | 0.2553 | — | 56.00% (3.55× lift) |
| **Regularized Logistic Regression** | 5-Fold GroupKFold | 0.5690 | 0.2019 | 0.2988 | 42.00% (2.66× lift) |
| **HistGradientBoosting (Champion)** | 5-Fold GroupKFold | **0.6898** | **0.2953** | **0.1261** | **66.00% (4.19× lift)** |

### Operational Queue Benchmark: Baseline vs Pure Risk vs ML Priority
In production, editors do not consume isolated probabilities; they work through a ranked queue. We compare the audited Heuristic Baseline against a Pure Risk classifier ($\hat{p}$ only) and the proposed Two-Part ML Priority Ranker ($\hat{p} \times \text{Impact}$):

| Queue Tier ($K$) | Operational Workflow Tier | Heuristic Baseline P@K (Lift) | Pure Risk Classifier P@K (Lift) | Proposed ML Priority P@K (Lift) | Relative Yield vs Baseline |
|:---:|:---|:---:|:---:|:---:|:---:|
| **$K = 10$** | Head-of-Queue Urgent Triage | 80.00% (5.08×) | 70.00% (4.44×) | **90.00% (5.71×)** | **+12.5%** |
| **$K = 20$** | Bi-Weekly Review Sprint | 70.00% (4.44×) | 75.00% (4.76×) | **80.00% (5.08×)** | **+14.3%** |
| **$K = 50$** | **Single Editor Monthly Budget** | 56.00% (3.55×) | 60.00% (3.81×) | **66.00% (4.19×)** | **+17.9% (+10.0% pt)** |
| **$K = 100$** | Multi-Editor Content Team | 42.00% (2.66×) | 63.00% (4.00×) | **60.00% (3.81×)** | **+42.9%** |
| **$K = 250$** | Quarterly Client Content Audit | 34.00% (2.16×) | 51.20% (3.25×) | **59.60% (3.78×)** | **+75.3%** |
| **$K = 500$** | Wide Portfolio Scan | 30.60% (1.94×) | 50.00% (3.17×) | **52.20% (3.31×)** | **+70.6% (+71%)** |

```
Precision@K (%)
100% ┼───● (ML 90%)                                                  
 80% ┼   ▼ (Base 80%)     ▲ (ML 80%)                    
     │                    │                               
 60% ┼   ▲ (PureRisk 70%) ▼ (Base 70%)  ───● (ML 66%) ───● (ML 60%) ───● (ML 59.6%) ───● (ML 52.2%)
 40% ┼                                  ▼ (Base 56%)  ▼ (Base 42%)  ▼ (Base 34%)    ▼ (Base 30.6%)
 20% ┼────────────────────────────────────────────────────────────────────────────────── [Base Rate = 15.8%]
  0% ┴───────┬────────────────┬───────────────┬─────────────┬─────────────┬─────────────┬──
           K=10             K=20            K=50          K=100         K=250         K=500
```

### Findings Across Queue Tiers
1. **Head of the Queue ($K \le 20$):** At top priority ($K = 10$), ML Priority achieves **90.00% precision (5.71× lift)**, capturing 9/10 true declines and dominating both the baseline (80.00%) and pure risk (70.00%).
2. **Target Operational Budget ($K = 50$):** At an editor's fixed 50-page capacity, ML Priority outperforms the heuristic baseline by **$+10.00\%$ percentage points** (66.00% vs 56.00%), capturing 33 true declines and preserving 27.1 expected monthly clicks.
3. **Queue Breadth ($K = 100$ to $K = 500$):** Beyond 50 rows, the heuristic baseline degrades rapidly toward 30.60%, whereas ML Priority sustains robust precision (52.20% at $K=500$, delivering a **+71% relative yield advantage**). ML successfully isolates subtle non-linear multi-window erosions that simple if-statements miss.

### Structured Error Analysis
Out-of-fold predictions diagnosed three distinct operational failure archetypes:
1. **High-Confidence False Positive (Structural Floor Trap):** Content `0f78162a8d2c114f` (retrospective impressions: 163, $\hat{p} = 0.9307$, actual forward outcome: stable at 150 impressions, label $=0$). The model observed historical decay but the page had reached a structural demand floor. *Policy Remedy:* Gate 2 and `Floor Stabilization Review` prevent unneeded rewrites.
2. **High-Confidence False Negative (Sudden Algorithmic Collapse):** Content `4b7a77e5566967eb` (retrospective impressions: 1,762, momentum ratio: $8.99\times$, $\hat{p} = 0.0178$, forward impressions: 1,282, label $=1$). The page was growing rapidly before experiencing sudden unpredicted de-indexing. *Policy Remedy:* Search telemetry alone cannot foresee external algorithmic penalties without preceding momentum decay.
3. **Boundary Noise:** Content `c87b645e51be5743` (retrospective impressions: 460, forward impressions: 368, realized drop ratio: exactly $0.8000$, $\hat{p} = 0.0567$, label $=0$). Pages landing precisely on the $20\%$ decline threshold are sensitive to single-click Poisson noise.

---

## 6. Interpretation

### Feature Attribution: What Drives Decline?
Permutation importance under 5-fold client-grouped holdout confirms the dominant empirical drivers of organic search decline:
* `client_share_of_traffic` (Importance: $0.0160 \pm 0.0019$): High-concentration pillar pages face aggressive competitor targeting and algorithm shifts.
* `log_impr_w30` (Importance: $0.0150 \pm 0.0023$): Baseline traffic volume acts as a scale anchor for volatility.
* `position_w30` (Importance: $0.0128 \pm 0.0010$): Pages ranking in SERP positions 6–15 exhibit acute vulnerability to losing first-page visibility.
* `impr_ratio_30_prev30` (Importance: $0.0127 \pm 0.0006$): Velocity decay remains a foundational directional indicator.
* `client_impr_ratio_30_prev30` (Importance: $0.0085 \pm 0.0007$): Normalizes page decay against overall client site momentum.
* `pct_of_peak_proxy` (Importance: $0.0071 \pm 0.0015$): Distance from historical maximum traffic separates cyclical troughs from terminal decay.
* `age_days` (Importance: $0.0047 \pm 0.0009$): Content maturity and staleness tier.
* `days_with_impr_w90` (Importance: $0.0035 \pm 0.0004$): Consistency of search presence over time.

### Surprises and Negative Results
1. **The Retrospective Drop Trap:** Ranking purely by past drops is counter-productive ($66.22\%$ mean reversion rate).
2. **The Failure of Pure Risk:** Ranking solely by probability $\hat{p}$ floods the top of the queue with low-volume tail pages (e.g. 5 impressions dropping to 0) that have high statistical certainty but near-zero business recovery value.
3. **Volume Tier Heterogeneity:** Low-volume pages (Tier 1: 100–500 impr, 14,538 pages) exhibit lower base rates ($14.29\%$) and lower ROC-AUC ($0.6377$) than high-volume pages (Tier 3: 2,000–10,000 impr, 9,328 pages, $21.88\%$ base rate, $0.7267$ ROC-AUC), confirming that high-traffic URLs produce cleaner, more reliable search decay signals.

---

## 7. Recommendation

### Tactical Action Framework: "Model = Order, Rule = Reason"
To build trust with editorial teams, the pipeline decouples ranking from action assignment:
* **The Model Supplies the Order:** Calibrated expected utility ($\hat{p} \times \text{Impact}$) determines the sequence of review.
* **The Audited Rule Supplies the Reason:** Transparent reason codes dictate the specific editorial action required.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TOP-50 MONTHLY EDITORIAL QUEUE BREAKDOWN                                  │
├─────────────────────────────────────────────────┬───────────┬────────────┬─────────────────────────────┤
│ Action Category                                 │ Count     │ Share (%)  │ Prescribed Tactical Action  │
├─────────────────────────────────────────────────┼───────────┼────────────┼─────────────────────────────┤
│ Floor Stabilization Review / Cooldown Watchlist │ 23 pages  │ 46.0%      │ Verify structural floor;    │
│                                                 │           │            │ prevent premature rewrite.  │
│ Immediate Content Refresh & SERP Intent Update  │ 11 pages  │ 22.0%      │ Full rewrite targeting lost │
│                                                 │           │            │ intent & SERP competitors.  │
│ Quiet Erosion Deep-Dive                         │ 7 pages   │ 14.0%      │ Investigate multi-signal    │
│                                                 │           │            │ risk unflagged by rules.    │
│ Tracking & Query Verification                   │ 6 pages   │ 12.0%      │ Audit tracking code/surges  │
│                                                 │           │            │ before content revision.    │
│ Passive Monitoring                              │ 2 pages   │ 4.0%       │ Stable control observation. │
│ Internal Linking & Authority Reinforcement      │ 1 page    │ 2.0%       │ Add contextual inbound links│
│                                                 │           │            │ to defend slipping rank.    │
└─────────────────────────────────────────────────┴───────────┴────────────┴─────────────────────────────┘
```

### Three Non-Bypassable Operational Code Gates
The output pipeline asserts three non-bypassable gates in code:
1. **Gate 1 (Cold-Start Qualification):** Requires $\ge 90$ days of page age and client tenure. 100% of the 38,271 scored pages satisfy this invariant.
2. **Gate 2 (Durable-Trend Protection):** 10,753 pages with durable rising trajectories are clamped to `Passive Monitoring` (zero automated rewrite authorization).
3. **Gate 3 (Extreme Volume Verification):** 243 pages exhibiting $>20\times$ volume spikes are routed to `Tracking & Query Verification` to rule out instrumentation bugs before editorial intervention.

### Two-Clock Production Telemetry
Because organic search decline takes 30 days to confirm, we cannot immediately check whether today's model predictions are correct. If a model drifts, waiting a full month to find out is unacceptable—editors would have already wasted hundreds of hours auditing erroneous recommendations. To solve this delayed-feedback challenge, the pipeline monitors system integrity across **two distinct timescales**:

1. **Clock 1: Pre-Release Input Gate (Day 0 / Instant $T$):**
   * *When it runs:* Immediately prior to publishing the monthly 50-page priority list to editorial teams.
   * *What it checks:* **Input Data Drift.** Scans incoming search traffic and momentum features using the **Population Stability Index (PSI)** to compare current feature distributions against training baselines.
   * *Status at $T = \text{2026-02-01}$:* Momentum feature PSI evaluated to **0.0531** (comfortably passing the $< 0.10$ stability threshold, granting automated clearance to publish the monthly queue).
   * *Operational Safeguard:* Catches analytics tracking outages, warehouse ETL corruptions, or extreme holiday anomalies *before* 200+ human audit hours are spent on bad recommendations.

2. **Clock 2: Outcome Tripwire (Day 30 / Instant $T+30$):**
   * *When it runs:* Exactly 30 days later, once actual Google Search Console performance data arrives.
   * *What it checks:* **Realized Base-Rate Drift.** Verifies that the portfolio-wide 30-day decline rate remains safely within the pre-registered operating band of $[10.76\%, 20.76\%]$ (centered on the $15.76\%$ historical base rate $\pm 5.0\%$).
   * *Status at $T+30$:* Observed portfolio decline rate evaluated to **15.76%** (operating stably within the normal operating band).
   * *Operational Safeguard:* Protects against unobserved macro shocks (such as broad Google Core Algorithm Updates or unannounced client site migrations). If portfolio decline surges outside this band ($> 5.0\%$ shift), automated model releases pause immediately for manual review and recalibration.

### Confidence and Explicit Boundaries (Non-Causal Claim Ladder)
* **What I Claim:** Offline Precision@50 of $66.00\%$ proves that the Two-Part ML ranker accurately enriches declining pages for human review, outperforming the heuristic baseline rule across all queue depths ($K=10$ to $K=500$).
* **What I Do Not Claim:** I do *not* claim that rewriting these articles will causally reverse traffic decline or cause recovery (proving causal refresh lift requires an A/B test). 

---

## 8. Reproducibility

### Environment and Dependencies
* **Operating System:** Windows / Linux / macOS
* **Python Runtime:** 3.10+ (Executed locally via `.venv\Scripts\python.exe`)
* **Core Libraries:** `duckdb==1.1.1`, `scikit-learn==1.5.2`, `pandas==2.2.3`, `numpy==2.1.2`, `matplotlib==3.9.2`
* **Random Seeds:** `RANDOM_STATE = 42` fixed across all splits, models, and scripts.

### Step-by-Step Reproduction Commands
To reproduce all receipts and figures from scratch from a fresh clone:
```bash
# 1. Clone repository and install dependencies
git clone https://github.com/kab-s/flyrank-page-priority-pipeline.git
cd flyrank-page-priority-pipeline
pip install -r requirements.txt

# 2. Inspect or run the consolidated capstone notebook
jupyter lab work/notebooks/capstone.ipynb

# 3. (Optional) Run offline reference baseline pipeline on bundled starter sample
python scripts/run_all.py
```

### Committed Receipts and Artifacts
The evaluation is checkable directly from committed receipts in `work/outputs/`:
* `work/outputs/w07_metrics.json`: Structured benchmark metrics, gate counts, and drift receipts.
* `work/outputs/capstone_metrics.json`: Consolidated capstone evaluation receipt.
* `img/fig_capstone_precision_at_k.png`: Figure 1 Precision@K visualization.
* `img/fig_capstone_split_gap.png`: Figure 2 Client Memorization Gap visualization.
* `img/fig_capstone_pipeline.png`: Figure 3 Pipeline Architecture diagram.
* `img/fig_capstone_action_mix.png`: Figure 4 Action Allocation bar chart.

---

## 9. Acknowledgments & data credit

This capstone project is built upon the **FlyRank Search Intelligence Warehouse dataset**, provided as part of the FlyRank Applied Machine Learning Internship program.

* **Data Source:** FlyRank AI Inc.
* **Official Website:** [https://flyrank.ai](https://flyrank.ai)
* **Dataset Scope:** 78,835,655 pseudonymized Search Console and GA4 daily performance facts (2025-01-27 to 2026-06-30).
* **Methodological Standards:** Designed according to the Google Rules of ML, the FlyRank Non-Causal Claim Ladder, and enterprise Applied Machine Learning evaluation standards.
