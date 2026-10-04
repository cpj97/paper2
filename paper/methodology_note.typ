#set document(title: "Methodological Approach: Protected Areas and Forest Loss")
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading: set text(weight: "bold")
#set table(stroke: 0.5pt)
#set figure(gap: 0.65em)

#align(center)[
  #text(16pt, weight: "bold")[Methodological Approach]

  #text(12pt)[Estimating the Impact of the Expansion of Protected Areas on Forest Loss in Colombia through a Staggered-Adoption Difference-in-Differences Design]

  #text(10pt, fill: rgb("#555555"))[Camilo Pedraza Jiménez, PhD Candidate, Hertie School]
]

#v(0.5em)

This note documents the empirical design implemented in my first research project: how the estimation sample is built for the staggered-adoption difference-in-differences design; which covariates and standard errors are used and when; the resulting average treatment effects on the treated (ATT); and the spatial robustness checks that motivate testing multiple control-group bandwidths. 

= Sample: Treated and Control Units

== Treated units

The treatment group is every 1km$times$1km grid cell that is added to a protected area (PA) for the first time during the study window. PA boundaries and designation years come from the World Database on Protected Areas (WDPA) @wdpa2024, pulled as yearly snapshots so a cell's treatment cohort reflects the designation year on record at that snapshot rather than the present-day boundary. The panel begins in 2001; treatment cohorts span 2002--2025 (no new cohort is observed in 2015--2017 or 2022--2023), giving 19 annual cohorts and 103,488 treated cells in total. This treated set is held fixed across every specification and robustness check in this note since it is not restricted to cells near a PA border. In this sense, the treatment effect being estimated is the effect of bringing a cell under protected-area status on its annual forest loss, averaged over the entire protected population rather than just cells at a PA's edge.

Cells that were already inside a PA before 2001 ("baseline" protection) are excluded from the regression sample entirely: they fail the treatment-cohort condition (their designation year does not exceed the panel start) and they are not eligible as controls either (they are not never-treated). They appear on the map in @fig:boundary-map purely for descriptive context, to make clear that the study-period cohorts expand an existing PA network rather than starting from zero.

== Control units and the boundary-band design

The never-treated cell pool is not homogeneous: forest-loss pressure varies systematically with proximity to a PA (documented in @sec:bandwidth below), so which never-treated cells are admissible as "clean" controls is itself a design choice.

- *Main buffer* (@sec:results): controls are never-treated cells whose distance to the nearest treated cell stays above a single 35km threshold for the entire panel. This threshold was established after the semivariogram evidence in @sec:bandwidth showed that spatial correlation in the estimator's own influence function does not reach its floor until roughly that distance.
- *Boundary-band robustness design* (@sec:bandwidth): controls for this robustness check are instead defined by a two-sided band of distance to the nearest already-designated PA: to its true polygon boundary for the near-border bands, and to the nearest treated cell for the band surrounding the main buffer directly. Six bands were defined for a reasonable comparison ([2,10], [2,15], [5,15], [5,25], [10,25], [25,35] km), plus the unbounded >35 km ring that is exactly the revised main specification's control threshold. Regarding the lower bound, a 2km minimum was chosen to avoid including cells that are may be inside a PA but misclassified as outside due to the 1km grid's coarse resolution.

@tbl:band-size shows the number of control cells available under each band (100,153 to 366,222 cells; treated cells are constant by construction at 103,488). The ordering is not monotonic in distance: >35km (264,952 cells) is smaller than the much narrower 5--25km ring (366,222 cells). This happens since the control pool size does not depend on the radius alone: a near-border band only requires a cell's distance to its single nearest PA to fall in a window, which much of the country satisfies given how many separate protected areas exist. However, the unbounded far ring requires being simultaneously far from every treated cell nationwide, a strictly harder condition once hundreds of protected areas are scattered across the landscape. @fig:boundary-map maps the geography of four bands (2--10km, 10--25km, 25--35km, and the unbounded >35km population) against the treated grid cells and the pre-2001 baseline PAs.

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.header([*Control band*], [*n control cells*]),
    [2--10km], [218,251],
    [2--15km], [318,954],
    [5--15km], [227,652],
    [5--25km], [366,222],
    [10--25km], [239,273],
    [25--35km], [100,153],
    [>35km (main buffer)], [264,952],
  ),
  caption: [Control-group size by boundary band],
) <tbl:band-size>

#figure(
  image("../outputs/figures/07_boundary_band_control_map_1km.png", width: 70%),
  caption: [Geography of four control bands against the treated grid cells and pre-2001 baseline PAs],
) <fig:boundary-map>

= Covariates

The doubly-robust specifications in @sec:results adjust for eight covariates: elevation, slope, and terrain ruggedness (derived in Earth Engine from the SRTM GL1 digital elevation model @farr2007shuttle), baseline forested area as of 2000 (the same Hansen Global Forest Change source @hansen2013high behind the forest-loss outcome variable for this research desings), pre-treatment DMSP-OLS and VIIRS nighttime-lights intensity @baugh2010development @elvidge2017viirs, and pre-treatment coca-cultivation intensity @unodc_simci, together with an explicit missingness indicator. For two of these, an imputation or missingness-handling decision was required, and the mechanism is documented here:

- *Nighttime lights*: DMSP-OLS is essentially complete until 2013, but VIIRS is missing for 10.4% of the analyzed sample, not at random. The VIIRS sensor only begins producing usable composites around 2012--2013, so VIIRS is missing for 100% of cells in every cohort designated 2002--2012 and 0% of cells in cohorts designated 2013 or later. Therefore,VIIRS is regression-imputed from DMSP (which covers the missing years) plus the geography covariates, fit by ordinary least squares on the cells where VIIRS is genuinely observed ($R^2=0.380$) and applied to the 48,778 cells where it is missing. Because the imputed values still vary with DMSP and geography cell by cell, it supplies real identifying variation for the early cohorts.
- *Coca-cultivation intensity*: this variable is the one candidate that is genuinely missing for most of the sample, since coca crops are present in specific areas of the country. Given existing evidence of a strong positive relationship between coca cropping and deforestation @marin2024coca, it is worth keeping this variable in the analysis. Therefore, missing values are zero-filled anda a "coca missing" indicator is added as a dummy variable indicating the absence of coca cropping rather than a source of sample loss.

The no-covariate specifications (`baseline`, `anticipation1`) use none of these covariates, while the doubly-robust specifications (`dr_geography`, `dr_geography_anticipation1`) use all eight together. The doubly-robust estimator of Sant'Anna & Zhao (#cite(<santanna2020doubly>, form: "year")) is implemented in the staggered difference in difference framework of Callaway & Sant'Anna (#cite(<callaway2021difference>, form: "year")).

= Standard Errors and Inference <sec:standard-errors>

The no-covariate specifications are estimated with the `csdid` Python package @csdid2024, while the doubly-robust specifications are instead estimated with an "in-house" doubly-robust plug-in estimator that allowed for a more flexible treatment of the eight covariates for the IPW estimation (regularization, cross-validation, and trimming). The two estimation tools are therefore not used interchangeably across specifications: `csdid` is only used for the no-covariate specifications, while the doubly-robust plug-in is only used for the doubly-robust specifications. 

Standard errors, by contrast, are computed in-house for all four specifications using the same Conley-style spatial-HAC (heteroskedasticity-and-autocorrelation-consistent) correction @conley1999gmm, which allows nearby cells' per-unit influence contributions $psi_i$ to be spatially correlated:
$
  hat("Var")(hat(tau)) = sum_i psi_i^2 + sum_(i != j, thin d_(i j) <= 35 "km") K(d_(i j) div 35 "km") thin psi_i psi_j
$
using a Bartlett kernel $K(dot)$, a 35km cutoff, and a streamed KD-tree search so the full pairwise distance matrix is never materialized. The no-covariate specifications use a plain 2$times$2 diff-in-means $psi_i$; the doubly-robust specifications use the doubly-robust plug-in $psi_i$ following Sant'Anna & Zhao (#cite(<santanna2020doubly>, form: "year")), which treats the fitted nuisance functions as known and so omits their Theorem 1 asymptotic correction terms. `csdid` does supply its own analytic SE as a byproduct for the no-covariate specifications, the analysis does not consider this type of standard error. the one exception is @tbl:analytic-vs-hac, which compares `baseline`'s analytic and spatial-HAC SEs directly to show how much the spatial correction alone changes inference. When looking at the RMS post-treatment SEs, the two are close for the main buffer once the buffer and cutoff were widened to clear the correlation floor identified in @sec:bandwidth. Thus, the residual spatial correlation left in the estimator's own errors barely matters here.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    table.header([*SE convention*], [*Assumption*], [*RMS post-treatment SE*#footnote[RMS-combined across post-treatment ATT$(g,t)$ cells as $sqrt(sum_i "SE"_i^2) \/ n$.]]),
    [Analytic (csdid)], [independence], [$25.89$],
    [Spatial-HAC], [allows spatial correlation], [$25.94$],
  ),
  caption: [Analytic vs. spatial-HAC standard error for the `baseline` specification],
) <tbl:analytic-vs-hac>

= Results <sec:results>

== Specifications and the role of anticipation

Four specifications are estimated, crossing two covariate choices with two anticipation assumptions:

#table(
  columns: (auto, auto, auto),
  align: (left, center, left),
  table.header([*Label*], [*Anticipation*], [*Covariates*]),
  [`baseline`], [0], [none],
  [`anticipation1`], [1], [none],
  [`dr_geography`], [0], [8 covariates (geography + nighttime lights + coca), doubly robust],
  [`dr_geography_anticipation1`], [1], [8 covariates (geography + nighttime lights + coca), doubly robust],
)

The `anticipation` parameter shifts the reference (pre-treatment base) period used for every ATT$(g,t)$ comparison one year earlier, from $g-1$ to $g-2$. Standard staggered-adoption estimators assume no anticipation: that treatment has zero effect before its recorded start date. In this setting, formal PA designation is typically preceded by public announcement, boundary demarcation, and early enforcement presence, so land users may already change behavior in the year immediately before the recorded designation year. Setting anticipation$=1$ avoids folding that anticipatory year into the "clean" pre-period baseline. Mechanically, this also means every post-period estimate is measured against a base period one year earlier than in the anticipation$=0$ version, so it integrates one additional year of treated-control divergence and will tend to be larger in magnitude even absent a true anticipation effect.

== General (overall) ATT

#figure(
  table(
    columns: (14em, auto, auto, auto),
    align: (left, right, right, center),
    table.header([*Specification*], [*ATT (m²/cell-year)*], [*ATT (% of control mean)*#footnote[Control mean is the mean annual loss among main-buffer control cells, pooled across the full panel ($1306.5$ m²/cell-year). Each percentage is the ATT divided by this control mean, times 100.]], [*Significance*#footnote[Significance is based on the spatial-HAC standard error, with $p<0.001$ (\*\*\*), $p<0.01$ (\*\*), $p<0.05$ (\*), and $p>=0.05$ (n.s.).]]),

    [`baseline`], [$-178.3$], [$-13.6%$], [\*\*\*],
    [`anticipation1`], [$-339.4$], [$-26.0%$], [\*\*\*],
    [`dr_geography`], [$-233.9$], [$-17.9%$], [\*\*\*],
    [`dr_geography_anticipation1`], [$-650.4$], [$-49.8%$], [\*\*\*],
  ),
  caption: [Overall ATT by specification for the main 35km-buffer sample],
) <tbl:overall-att>

The no-covariate specifications point to a robust, negative effect of protection on annual forest loss: `baseline` estimates a reduction of about 178 m² per treated cell-year, rising to 339 m² under `anticipation1`, consistent with the mechanical amplification described above. Both are estimated with high significance under the spatial-HAC correction. The doubly-robust specifications, estimated with the per-cohort-pooled, regularized, trimmed plug-in described in @sec:standard-errors, produced the following estimates: `dr_geography` estimates $-233.9$ and `dr_geography_anticipation1` estimates $-650.4$. Both have the same sign as their no-covariate counterparts, and are significantly estimated.

In percentage terms, `baseline` indicates that bringing a cell under protected-area status reduces its annual forest loss by 13.6%, relative to the rate observed on comparable unprotected land, rising to 26.0% under `anticipation1`. The doubly-robust specifications are larger in magnitude: `dr_geography` indicates a 17.9% reduction, and `dr_geography_anticipation1` indicates a nearly 50% reduction, both relative to the same control mean. The anticipation assumption matches the expected direction: the ATT is larger in magnitude when the pre-period baseline is shifted one year earlier, consistent with anticipatory behavior mentioned in the previous section.

#figure(
  image("../outputs/figures/08_boundary_band_overall_att_1km.png", width: 78%),
  caption: [Overall ATT by control band and specification],
) <fig:overall-att-band>

@fig:overall-att-band shows the estimates for all four specifications against the two-sided boundary-band control definitions introduced in @sec:bandwidth, holding the treated set fixed throughout, and adds the main >35km buffer as a reference. The no-covariate specifications show the gradient already documented for `baseline` alone in @fig:leakage-grid from @sec:bandwidth: the point estimate moves from small and only marginally significant near the border (2--10km: `baseline` $+58.6$, $p=0.093$) to increasingly negative and significant farther out (10--25km: $-135.2$, $p<0.001$), landing on $-178.3$ in the main buffer. This finding suggests that the control band's distance to the border matters in the expected direction.

The doubly-robust specifications tell an additional story across every near-border band. From 2--10km through 25--35km, both `dr_geography` and `dr_geography_anticipation1` are mostly positive, showing the opposite sign from the no-covariate specifications in the same bands. Additionally, the confidence intervals for the near-border bands are much wider than the ones for the main buffer. Only once the control pool moves to the clean main buffer do the doubly-robust point estimates flip to match the no-covariate specifications' sign and lead to more robust confidence intervals.

The previous finding matches the mechanism discussed in @sec:bandwidth: near the border, the covariates the doubly-robust estimator relies on to adjust for confounding (for instance, nightlights) carry a different, locally reversed relationship to treatment than they do over the full sample, while the raw near-border signal is itself leakage-contaminated. Therefore, the covariate adjustment there is not cleanly removing confounding, but reacting to a less stable local relationship. The boundary-band grid is therefore best read as a *leakage diagnostic*: a near-border comparison can flip sign once adjusted for covariates, which is itself evidence against trusting it. The main 35km buffer, where the covariate-adjusted and unadjusted estimates agree, is the specification worth considering for robust results.

== Event-study ATT

#figure(
  image("../outputs/figures/08_boundary_band_event_study_1km.png", width: 92%),
  caption: [Event-study ATT by control boundary band and specification],
) <fig:event-study-band>

@fig:event-study-band shows the event-study ATT for all four specifications by boundary band, rather than the single overall ATT. Pre-treatment event times ($t<0$) are nearly indistinguishable across bands within a given specification. These pre-trend estimates oscillate within a few hundred m² of zero without a clear one-directional drift, wich can serve as visual evidence for the parallel-trends assumption that underlies the staggered-adoption design. 

When looking at post-treatment ($t>=0$) effects, the near-border bands tend to sit visibly higher (more positive, or less negative) than the far bands at most event times, for all four specifications. `baseline` is mostly non-significant across every band and event time. `anticipation1`, by contrast, turns clearly negative, consistent with its larger anticipation-adjusted overall ATT mentioned before. The doubly-robust specifications show the opposite pattern near the border: `dr_geography` and `dr_geography_anticipation1` are positive or non-significant across the near-border bands and only turn negative once the control band reaches the clean main buffer, which is consitent with the sign-flip-with-distance pattern already described for the overall ATT. Standard errors are also visibly smaller at the main buffer than at any near-border band, and including covariates further reduces the near-border bands' sensitivity to outlier cohorts, such as the cohort-2013 spike at post-treatment event time $3$ (@sec:cohort-2013-outlier).

== Cohort-level heterogeneity in the event study <sec:cohort-heterogeneity>

The event-study panels above pool every cohort together at each event time, which can hide cohort-specific behavior behind a simple average. @fig:cohort-parallel-trends-full plots each of the 19 cohorts' own ATT$(g,t)$ trajectory directly across the full event window, main buffer, all four specifications. Each line represents a cohort, with darker colors indicating later cohort years and line width encoding cohort size. The cohort-size-weighted mean is shown as a plain dot at each event time, with an ordinary-least-squares trend fit separately to the pre-period and post-period dots, so the pre-trend and post-treatment slopes can be compared directly.

The left panel, with its whole-scale, unclipped view, shows the full range of cohort-event estimates, making the outlier at post-treatment event time $3$ for cohort 2013 clearly visible. The right panel zooms in to $plus.minus 2000$ m² for a closer read of the cohorts' trajectories. The pre-period fit stays close to zero for all specifications, while the post-period fit sits mostly below zero, mirroring the negative overall ATT in @tbl:overall-att. Individual cohorts oscillate around zero in the pre-period without a clear one-directional drift, and so do the weighted means. These last two findings are consistent with the parallel-trends assumption.

#figure(
  image("../outputs/figures/08_diffdiff_cohort_parallel_trends_full_combined_1km.png", width: 100%),
  caption: [Cohort ATT, full event window, with separate pre/post trend fits (main buffer)],
) <fig:cohort-parallel-trends-full>

@tbl:cohort-weight reports each cohort's share of the cohort-size weighting separately for the pre-period ($t<=-1$) and post-period ($t>=0$) weighted means. Cohort size is fixed, but each cohort's influence on the weighted mean still varies by event time, since not every cohort has an estimate at every event time. Cohort 2002, for instance, was designated one year after the panel start, so 2001 (its only possible pre-treatment year) is also its own base period $g-1$, which by construction is never assigned its own ATT$(g,t)$ estimate. It therefore contributes zero pre-period estimates despite being a moderately large cohort (11,294 treated cells), so its weight in the table below is exactly 0.0% pre-treatment versus 10.9% post-treatment. The set of cohorts behind a pre-period weighted mean is therefore not automatically the same set, with the same relative weights, behind a post-treatment one.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Cohort*], [*n cells*], [*Share of weight (pre)*], [*Share of weight (post)*]),
    [2002], [11,294], [0.0%], [10.9%],
    [2003], [93], [0.1%], [0.1%],
    [2004], [41], [0.0%], [0.0%],
    [2005], [1,104], [1.2%], [1.1%],
    [2006], [773], [0.8%], [0.7%],
    [2007], [4,207], [4.6%], [4.1%],
    [2008], [2,662], [2.9%], [2.6%],
    [2009], [14,394], [15.6%], [13.9%],
    [2010], [1,118], [1.2%], [1.1%],
    [2011], [12,195], [13.2%], [11.8%],
    [2012], [897], [1.0%], [0.9%],
    [2013], [1,730], [1.9%], [1.7%],
    [2014], [368], [0.4%], [0.4%],
    [2018], [11,890], [12.9%], [11.5%],
    [2019], [9], [0.0%], [0.0%],
    [2020], [24,879], [27.0%], [24.0%],
    [2021], [2,119], [2.3%], [2.0%],
    [2024], [10,865], [11.8%], [10.5%],
    [2025], [2,850], [3.1%], [2.8%],
  ),
  caption: [Approximate cohort weight in the size-weighted mean, by pre- and post-treatment window],
) <tbl:cohort-weight>

== Outlier diagnostic: cohort 2013 <sec:cohort-2013-outlier>

The previous section allowed the detection of a clear outlier: cohort 2013 at event time $3$ (calendar year 2016) reports ATT over 20,000 m² across all four specifications, far higher than any other (cohort, event-time) combination. Cohort 2013 carries only 1.7% of the weight in the cohort-weighted overall ATT (@tbl:cohort-weight), but the outlier is large enough to visibly distort simple, unweighted per-event-time averages, including the pooled event-study panels in @fig:event-study-band and the within-cohort figures above.

@fig:cohort-2013-diagnostic traces this spike to its source rather than leaving it as an unexplained number. Of the cohort's 1,730 cells, only 300 have any recorded loss in 2016 at all (the median is zero), and only 106 of these cells lost more than 100,000 m² that year. Those 106 yet account for 93.3% of the cohort's total 2016 loss. Those 106 "driver" cells can be found inside and around the *Lago Azul Los Manatíes* PA, whose designation of 2013 matches this analyzed cohort. Splitting the cohort's raw annual loss trajectory into driver and non-driver cells shows that the non-driver cells follow the trend of the the never-treated control pool closely throughout 2001--2025, while the spike is driven when addint the driver subset.

#figure(
  image("../outputs/figures/08_cohort_2013_outlier_diagnostic_1km.png", width: 95%),
  caption: [Outlier diagnostic for cohort 2013],
) <fig:cohort-2013-diagnostic>

This location and timing match a well-documented, independently reported event rather than a data anomaly: Colombia's national deforestation statistics for 2016 show a 44% increase over 2015, and specifically attribute the single largest fire-driven deforestation area detected in the country since 1992 to a March 2016 fire on the Antioquia--Chocó border, amid a wider wave of forest fires in Antioquia during that year's El Niño-driven dry season @semana2017deforestacion. The Antioquia/Chocó border location, home to Lago Azul Los Manatíes, and the year (2016) match precisely the location and timing of the driver cells identified above. The spike is therefore read as a real, externally corroborated clearing event concentrated in a small number of cells and not product of the estimation.

= Bandwidth Sensitivity and Spatial Evidence <sec:bandwidth>

The specifications above hold the control definition fixed. This section asks a different question: does the estimated effect depend on how far from a PA border the comparison group is allowed to sit? Three pieces of evidence help answer this.

*Where deforestation pressure sits, cross-sectionally.* @fig:level-profile plots raw mean annual loss as a function of signed distance to the nearest PA's true polygon boundary (negative = inside, positive = outside), pooling both treated and never-treated cells. Loss rises steadily from a low, flat level deep inside a park to about 1,240 m²/year right at the edge, then jumps sharply to roughly 1,700 m²/year just outside, stabilizing across the exterior range. This is a purely descriptive snapshot (it mixes pre- and post-treatment years for treated cells), but it establishes that the PA border is a real structural discontinuity in the landscape, not an arbitrary line.

#figure(
  image("../outputs/figures/07_boundary_level_profile_1km.png", width: 92%),
  caption: [Loss level by distance to nearest PA boundary (deep interior to far exterior)],
) <fig:level-profile>

*Whether that pressure contaminates the control pool.* @fig:variogram plots the empirical semivariogram#footnote[A (semi)variogram is the key function used to fit a model of the spatial correlation of an observed phenomenon @bachmaier2011variogram.] of pre-treatment forest-loss levels against pairwise distance between grid cells, together with a fitted spherical model @cressie1993statistics. The curve climbs to a local peak of about 12.6 million around 37km, then decreases to about 11.9 million by 53km, and climbs back after 100km, to about 14.1 million by 135km.

A spherical model is additionally fit to gain a more solid understanding of the main buffer used in the analysis. However, this model can only describe a curve that rises once and then settles at a flat level, which is why the fit is restricted to a lag of 40km, since the semivariance peaks around that point. This model gives three numbers that summarize the rise: a nugget of about 8.92 million (the leftover variation between cells even at very short distances), a sill of about 12.58 million (the level the curve settles at), and a range of about 31.9km (the distance up to which nearby cells still share enough local structure to be correlated), with a good fit ($R^2=0.95$). The later decrease and rise of the semivariance could reflect that the different regions of Colombia (Andes, Amazon, Caribbean, Pacific, Orinoquia) simply sit at different average loss levels, not that nearby cells become more or less alike with distance. 

The fitted $approx 32$km range, together with the local peak around 37km of the semivariance, is the empirical basis for the defined main buffer of 35km threshold above: below that distance, nearby treated and control cells' outcomes are still measurably correlated. This fact can serve as evidence of leakage that would make a "clean" control definition drawn any tighter than this unreliable. 

#figure(
  image("../outputs/figures/07_variograms_1km.png", width: 68%),
  caption: [Empirical semivariogram of pre-treatment forest-loss],
) <fig:variogram>

@fig:hajek-profile#footnote[Following the design-based, distance-indexed Hajek estimator of Wang et al. (#cite(<wang2020design>, form: "year")): each 1km distance bin's pre/post change is referenced against the far-exterior band's own pre/post change, computed separately per cohort and event time, compensating for shocks shared with that reference.] complements this with a distance-indexed, referenced contrast in the spirit of Wang et al. (#cite(<wang2020design>, form: "year")): rather than a single fixed reference band, each 1km bin's pre/post change is benchmarked against a far-exterior reference computed separately per cohort and event time. This reference band was established at $[35,50]$km as it sits just after the main buffer of 35km.

Unlike the boundary-band grid's single ATT per discrete band, this approach traces the pre/post change in loss *continuously* by distance to the border. The reference band ($[35,50]$km) sits at essentially zero, as it must, confirming the far-exterior baseline is well-behaved. Additionally, the curve outside the border converges to zero by roughly 10--15km and stays flat from there, meaning whatever is happening near the border has decayed well before the 35km buffer, corroborating the conclusion the semivariogram reached from raw spatial correlation.

However, it is important to highlight that the convergence to zero just outside the border is not monotonic: there is a positive jump just outside the border, then a slightly negative dip, before converging to zero at around 20km. Additionally, there is more noise inside the border, with more variance and wider confidence intervals. This figure does not resolve leakage versus frontier selection, but confirms that the near-border zone differs from the far reference on both sides and that this difference fully decays before 35km. This approach is mostly descriptive by design, so it serves as a diagnostic alongside the boundary-band grid rather than an additional headline estimate.

#figure(
  image("../outputs/figures/09_boundary_hajek_distance_profile_1km.png", width: 55%),
  caption: [Boundary effect vs. far exterior reference(Hajek-style, distance-indexed)],
) <fig:hajek-profile>

*Whether it actually moves the ATT.* @fig:leakage-grid reports the boundary-band robustness grid directly with no covariates or anticipation, as a sanity check. @fig:overall-att-band and @fig:event-study-band (in @sec:results) show the same grid extended to all four specifications, overall and by event time respectively. The pattern trends in the expected direction without being perfectly monotonic: as the control band moves farther from the border, the estimated effect goes from a small, only marginally significant positive gap (2--10km: $+58.6$, $p=0.093$) to an increasingly large and significant negative effect (10--25km: $-135.2$, $p<0.001$), before attenuating again at 25--35km ($-81.1$, n.s.). This band has the smallest control pool of the six bands, which widens the SE enough to lose significance. The main >35km buffer continues the negative trend at $-178.3$ (@tbl:overall-att).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, center),
    table.header([*Control band*], [*n control cells*], [*ATT*], [*SE (spatial-HAC)*], [*Sig.*]),
    [2--10km], [218,251], [$+58.6$], [$34.9$], [.],
    [2--15km], [318,954], [$+11.8$], [$26.5$], [n.s.],
    [5--15km], [227,652], [$-26.4$], [$27.2$], [n.s.],
    [5--25km], [366,222], [$-87.5$], [$26.1$], [\*\*\*],
    [10--25km], [239,273], [$-135.2$], [$26.7$], [\*\*\*],
    [25--35km], [100,153], [$-81.1$], [$79.8$], [n.s.],
  ),
  caption: [Overall ATT at the baseline specification by control band, spatial-HAC SE],
) <fig:leakage-grid>

By reading the different pieces of evidence together, it is possible to see that the PA border is a genuine structural discontinuity, that spatial correlation in the estimator's own error extends to a fitted range of roughly 35km before stabilizing, and that contamination is large enough to materially shift the estimated ATT, depending on where the control band is drawn. This evidence shows that control cells near the border behave differently, but not why. Two mechanisms could explain these results: *genuine leakage*, where protection pushes clearing activity into unprotected land just across the border, so a nearby "control" cell is itself affected by treatment; and *frontier selection*, where protected areas simply tend to be placed at the edge of accessibility frontiers @andam2008measuring @joppa2009high, so cells near any PA border would already look different from far-away cells even if protection had no effect at all. Due to these reasons, the main specification deliberately uses a 35km buffer to prevent leakage and additional covariates that can be useful for adjusting for frontier selection. 

#pagebreak()
#bibliography("references/methodology.bib", title: "References", style: "apa")
