# Diagnostics of distributional misspecification in HANK models

Code for the M.Sc. thesis of Francesco Pinna, Department of Economics and Finance, LUISS Guido Carli.

## Summary

**Question.** Heterogeneous-agent New Keynesian (HANK) models are commonly evaluated on their ability to reproduce the response of aggregate consumption to a monetary policy shock. As Werning (2015) and Acharya and Dogra (2020) show, the conclusions of these models depend on how the uninsured income risk of households moves with aggregate conditions. The thesis asks whether the responses observed after a monetary policy shock can reveal that a model describes this risk incorrectly, and for which households.

**Purpose.** The aim is a measure of how far a given model is from the data in this respect, expressed as the smallest change in the income process of households that reconciles the model with the observed responses, together with the part of the wealth distribution in which the change is needed. Such a measure tells the user of a model how much to trust its distributional conclusions, which are the ones that distinguish it from a representative-agent model.

**Method.** The alternatives to a solved reference model are the models that coincide with it except for the transition probabilities of household income in the quarters after the shock, which may differ across wealth groups. Among the alternatives that reproduce the observed responses, the method selects the one closest to the reference, with closeness measured by relative entropy, and reports its distance and the wealth groups in which the probabilities differ.

**State of the work.** On data simulated from the model, the response of aggregate consumption is not enough for the method, because it does not distinguish a change in income risk from the monetary impulse. Adding the consumption of the poorer half of households separates the two and identifies up to two wealth groups. Adding the income expectations of households by group, which the ECB Consumer Expectations Survey collects, identifies up to four wealth groups and tells a change in risk from a change in expected income. With expectations, the consumption of the poorer half, for which the euro area has no public source, is no longer needed. The application to euro-area data is the next step.

## 1. Reference model

Time is discrete and a period is a quarter. The economy is populated by a continuum of households of measure one, a government and a monetary authority, and has no aggregate uncertainty. It rests in its steady state until quarter 0, when an unanticipated monetary policy shock occurs, and from then on households know the path of every aggregate variable.

### Households

A household enters quarter $`t`$ with $`b_t`$ units of a one-period real bond and with an earnings state $`e_t`$. The earnings state takes one of $`n`$ values and follows a Markov chain with transition probabilities $`p(e' \mid e)`$, independent across households. Markets are incomplete, since the bond is the only asset and borrowing is not allowed. The household chooses consumption $`c_t`$ and bond holdings $`b_{t+1}`$ to solve

```math
\begin{aligned}
&\max \;\; \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^{t}\, \frac{c_t^{\,1-\gamma}}{1-\gamma} \\[4pt]
&\text{subject to} \quad c_t + b_{t+1} = (1+r_t)\, b_t + \frac{e_t}{\bar e_t}\,(Y_t - T_t), \qquad b_{t+1} \ge 0 .
\end{aligned}
```

Here $`\beta`$ is the discount factor, $`\gamma`$ the coefficient of relative risk aversion, $`r_t`$ the real interest rate on bonds held from quarter $`t-1`$ to quarter $`t`$, $`Y_t`$ aggregate output, $`T_t`$ aggregate taxes and $`\bar e_t`$ the average earnings state across households. Labour is supplied inelastically, and each household receives the share $`e_t / \bar e_t`$ of aggregate after-tax income, so that the earnings state governs how income is divided among households while total income is determined in equilibrium. The optimal decisions $`c_t(b,e)`$ and $`b_{t+1}(b,e)`$ satisfy the Euler equation

```math
c_t(b,e)^{-\gamma} \;\ge\; \beta\,(1+r_{t+1}) \sum_{e'} p(e' \mid e)\; c_{t+1}\bigl(b_{t+1}(b,e),\, e'\bigr)^{-\gamma},
```

with equality whenever $`b_{t+1}(b,e) > 0`$.

Let $`\lambda_t(b,e)`$ denote the share of households that enter quarter $`t`$ with bonds $`b`$ and earnings state $`e`$. Aggregate consumption and average earnings are

```math
C_t = \sum_{b,e} \lambda_t(b,e)\, c_t(b,e), \qquad \bar e_t = \sum_{b,e} \lambda_t(b,e)\, e ,
```

and $`\bar e_t = 1`$ at all dates in the reference model.

### Prices, government and monetary authority

Output is produced with labour and is determined by demand. Prices are sticky, and inflation $`\pi_t`$, the growth rate of the price level between quarters $`t-1`$ and $`t`$, follows a New Keynesian Phillips curve

```math
\pi_t = \kappa \,\log\frac{Y_t}{\bar Y} + \beta\, \pi_{t+1},
```

where $`\bar Y`$ is steady-state output. Because labour supply is inelastic, the curve is imposed in reduced form, with its slope $`\kappa`$ as a parameter.

The government keeps the stock of bonds constant at $`B`$ and pays the interest with taxes, $`T_t = r_t B`$. Taxes are levied in proportion to earnings and are not distortionary.

The monetary authority sets the nominal interest rate $`i_t`$ on bonds held from quarter $`t`$ to quarter $`t+1`$ according to the rule

```math
i_t = \bar r + \phi\, \pi_t + v_t, \qquad r_{t+1} = i_t - \pi_{t+1},
```

where $`\bar r`$ is the steady-state real rate, $`\phi`$ the response to inflation and $`v_t`$ the monetary policy shock, and the second equation is the Fisher relation. The return $`r_0`$ equals $`\bar r`$, because the bonds held when the shock occurs were issued before it.

### Equilibrium

Given a path of the shock $`v_t`$, an equilibrium consists of paths of $`r_t`$, $`i_t`$, $`\pi_t`$, $`Y_t`$ and $`T_t`$, of decision rules $`c_t(b,e)`$ and $`b_{t+1}(b,e)`$ and of distributions $`\lambda_t`$ such that households optimise, the distribution evolves according to the decision rules and the transition probabilities, the Phillips curve, the policy rule, the Fisher relation and the government budget hold, and the goods market clears,

```math
C_t = Y_t .
```

The bond market then clears by Walras' law,

```math
\sum_{b,e} \lambda_t(b,e)\, b_{t+1}(b,e) = B .
```

The steady state is the equilibrium with $`v_t = 0`$ at all dates, in which aggregate after-tax income is normalised to one.

### Parameter values

Three parameterisations share $`\gamma = 2`$, $`\bar r = 0.01`$ per quarter, $`\kappa = 0.05`$ and $`\phi = 1.5`$.

| | Two-state example | Seven states, low wealth | Seven states, high wealth |
|---|---|---|---|
| Name in the code | `lab` | `liquid` | `total` |
| Earnings state | 0.70 or 1.30, with probability 0.075 of changing state in a quarter | logarithm follows an AR(1) with autocorrelation 0.966 and standard deviation 0.5, on seven states | as in the previous column |
| Discount factor $`\beta`$ | 0.970 | 0.954 | 0.975 |
| Bond supply $`B`$, in quarters of average income | 1.51 | 1.04 | 5.60 |
| Households at the borrowing limit | 19 percent | 29 percent | 13 percent |

The two-state example is deliberately simple and serves to understand the mechanisms, with a quarterly marginal propensity to consume out of a transfer of 0.80 in the poorest quartile of the wealth distribution and of 0.05 in the richest. In the other two parameterisations the AR(1) process has the parameters of McKay, Nakamura and Steinsson (2016) and is discretised on seven states with the method of Rouwenhorst (1995), the high-wealth bond supply follows their calibration, and the low-wealth bond supply equals the liquid assets of households in Kaplan, Moll and Violante (2018), 0.26 of annual output. In both, the discount factor is chosen to match the bond supply. With low wealth, 29 percent of households are at the borrowing limit, close to the 30 percent of households without liquid wealth that Kaplan, Moll and Violante report for the United States.

### Monetary policy shock

The shock used throughout is a cut of the nominal rate of 25 basis points per quarter, one percentage point at an annual rate, that decays at rate 0.8 per quarter, $`v_t = -0.0025 \times 0.8^{t}`$. In the two-state example consumption rises by 0.44 percent on impact, inflation by 11 basis points, and the real rate between quarters 0 and 1 falls by 18 basis points.

## 2. Alternative models

An alternative model has the same preferences, policy rule, bond supply and steady state as the reference model, and differs from it only in the transition probabilities of the earnings state. For 40 quarters from quarter 0 these may deviate from $`p`$, by an amount that depends on the wealth of the household and on the date, after which they return to $`p`$. In the reference model a monetary policy shock leaves the income process unchanged, and the alternatives are therefore models in which the income process moves after the shock, possibly in different ways along the wealth distribution.

Households are divided into $`J`$ groups of equal size according to their bond holdings, with $`J`$ equal to 1, 2, 4 or 8. With $`J = 4`$ the groups are the quartiles of the steady-state wealth distribution, except that the households at the borrowing limit are kept in the same group. For a household that leaves quarter $`t`$ with bond holdings in group $`j`$, the probability of moving from $`e`$ to a different state $`e'`$ is

```math
\tilde p_t(e' \mid e, j) \;=\; p(e' \mid e)\,\bigl(1+\Delta^{\mathrm{risk}}_{j,t}\bigr)\times
\begin{cases}
1+\Delta^{\mathrm{mean}}_{j,t} & \text{if } e' > e,\\[2pt]
1-\Delta^{\mathrm{mean}}_{j,t} & \text{if } e' < e,
\end{cases}
```

and the probability of remaining in $`e`$ is such that the probabilities sum to one.

- A change in income risk, $`\Delta^{\mathrm{risk}}_{j,t} > 0`$, makes every change of earnings state more likely, upwards and downwards alike, so that earnings become less persistent and next quarter's earnings more uncertain.
- A change in expected income growth, $`\Delta^{\mathrm{mean}}_{j,t} > 0`$, makes upward moves more likely and downward moves less likely.

A change in risk common to all groups leaves the distribution of earnings unchanged for the economy as a whole. Within a wealth group it also moves expected income, because earnings return faster towards their average, and expected income therefore rises in the poorest group, where low earnings prevail. The two types of change remain distinct, since a change in risk moves the expected earnings of households with low earnings and of those with high earnings in opposite directions, whereas a change in expected income growth moves them in the same direction.

Households know the path of these probabilities from quarter 0 and decide optimally given them, taking the probabilities of their own group as given. The Euler equation holds with $`\tilde p_t`$ in place of $`p`$, and the equilibrium conditions of Section 1 are unchanged. An alternative model is thus a perfect-foresight equilibrium of the same economy with a different income process. Because household income is the share $`e_t / \bar e_t`$ of aggregate income, a change of the income process redistributes income across households and leaves the determination of aggregate income to the equilibrium conditions.

Errors of specification of another kind, for instance in the borrowing limit, lie outside this class and are to be studied by changing the reference model.

## 3. Distance between an alternative and the reference

The distance of an alternative from the reference is the relative entropy (Kullback-Leibler divergence) of its income process with respect to the reference one, per household, cumulated over the quarters after the shock,

```math
\mathcal{D}(\tilde p \,\|\, p) \;=\; \sum_{t \ge 0}\; \sum_{j,\,e} \tilde\lambda_t(j,e) \sum_{e'} \tilde p_t(e' \mid e, j)\, \log\frac{\tilde p_t(e' \mid e, j)}{p(e' \mid e)} ,
```

where $`\tilde\lambda_t(j,e)`$ is the share of households that leave quarter $`t`$ in wealth group $`j`$ with earnings state $`e`$. Relative entropy is the expected logarithm of the likelihood ratio between the two models, and it measures how easily they can be told apart by observing the income histories of households. Hansen and Sargent (2008) translate it into a detection error probability, the probability of choosing the wrong model on the basis of the data. With $`N`$ independent household histories this probability is approximately $`\Phi\bigl(-\sqrt{N \mathcal{D}/2}\bigr)`$, where $`\Phi`$ is the standard normal distribution function. It equals one half when the two models are indistinguishable and falls towards zero as the distance grows.

For small deviations the distance is approximately quadratic,

```math
\mathcal{D}(\tilde p \,\|\, p) \;\approx\; \frac{1}{2} \sum_{t \ge 0} \sum_{j} \omega_j \Bigl[\bigl(\Delta^{\mathrm{risk}}_{j,t}\bigr)^2 + \bigl(\Delta^{\mathrm{mean}}_{j,t}\bigr)^2\Bigr],
```

where the weight $`\omega_j`$ is the number of changes of earnings state that originate in group $`j`$ in a quarter of the steady state, per household in the economy. A given percentage deviation is therefore farther from the reference in a group in which many changes of earnings state take place.

## 4. Method

The data are a vector $`\hat x`$ of responses to a monetary policy shock, estimated with error. The covariance matrix of the estimates, with one row and one column for each response, is written $`\mathrm{Var}(\hat x)`$. In the code the vector contains, in percentage deviation from the steady state,

- the response of aggregate consumption over 24 quarters,
- where the experiment includes it, the response of the consumption of the poorer half of households, those with bond holdings below the steady-state median, over 12 quarters,
- where the experiment includes them, the responses over 12 quarters of expected income one year ahead and of its standard deviation, averaged within groups of households.

Let $`x(\Delta, v)`$ denote the same vector computed in the equilibrium of the model with deviations $`\Delta`$ of the income process and path $`v`$ of the monetary policy shock, and let $`\hat v`$ be the path of the shock assumed by the researcher. The method solves

```math
\begin{aligned}
\min_{\Delta,\, v} \;\; & N\, \mathcal{D}\bigl(\tilde p(\Delta) \,\|\, p\bigr)
\;+\; \frac{1}{2} \sum_{t \ge 0} \Bigl(\frac{v_t - \hat v_t}{\sigma_v}\Bigr)^{2} \\[4pt]
& +\; \frac{1}{2}\, \bigl(\hat x - x(\Delta, v)\bigr)'\, \mathrm{Var}(\hat x)^{-1}\, \bigl(\hat x - x(\Delta, v)\bigr).
\end{aligned}
```

The first term is the distance of the alternative from the reference, multiplied by the number $`N`$ of independent household histories from which the two models are to be told apart. A larger $`N`$ makes deviations of the income process more costly relative to the fit of the responses. It is set to 8,000 in the experiments, and in an application it is a parameter to be fixed in advance (Section 7). The second term allows the monetary impulse to differ from the assumed one, at a cost governed by $`\sigma_v`$, set to 25 basis points. It represents an aggregate explanation of a gap between model and data, which competes with the distributional explanations. The third term penalises the part of the observed responses that the alternative does not reproduce, according to the precision of the estimates.

The method reports the minimised distance, with its detection error probability, and the division of the minimised objective among the monetary term, the wealth groups and the two types of change, which indicates where the model needs to be corrected. Since the closest alternative is selected, the distance is a lower bound, in the sense that no alternative closer to the reference reproduces the data while the true model may be farther. The statement holds when the responses are measured without error, and only approximately when they are estimated, because the selected alternative then also reproduces part of the estimation error. The source indicated by the method is the component with the largest share, after dividing each share by the largest value it can take for any data, a normalisation that prevents the components with a large effect on consumption from being selected by construction.

For the computation, the map $`x(\Delta, v)`$ is linearised around the reference equilibrium. Its matrix of derivatives with respect to changes announced for each future quarter is computed numerically, following the sequence-space approach of Auclert, Bardóczy, Rognlie and Straub (2021), and the distance is replaced by its quadratic approximation, so that the problem is quadratic and has a closed-form solution.

## 5. Experiments

Before turning to data, the method is tested on responses generated by the model itself, for which the correct answer is known. In each experiment a single disturbance is active, one of the following.

- the monetary policy shock of Section 1, or
- a change in income risk in one wealth group, $`\Delta^{\mathrm{risk}}_{j,t} = 0.15 \times 0.85^{t}`$, or
- a change in expected income growth in one wealth group, $`\Delta^{\mathrm{mean}}_{j,t} = 0.15 \times 0.85^{t}`$.

The equilibrium responses are computed without approximation and without estimation error and are handed to the method, which is asked to indicate the disturbance that generated them. An experiment is successful when the method indicates the correct source, wealth group and type of change, and its margin is the difference between the two largest normalised shares. The exercise is repeated for different sets of observed responses and for different numbers of wealth groups, in order to learn which observations are needed and how fine a division of the wealth distribution they support.

The table of Section 6 refers to the monetary policy shock and to a common variance of the responses, chosen so that the selected alternative reproduces on average 95 percent of them. As checks, the experiments with two types of change are repeated with a declared covariance matrix, which has standard errors of 0.2 percent for the consumption responses and 0.3 percent for the expectations and autocorrelation 0.9 across horizons, with a shock to the discount factor of households, $`\beta_t = \beta\,(1 + 0.005 \times 0.8^{t})`$, in place of the monetary policy shock, and with the responses net of the effect of the path of the real rate.

## 6. Results

In the table, a tick means that every experiment is successful, "group" means that the wealth group is always correct while the two types of change are confused, and a cross means that at least one experiment is assigned to the wrong source or group. The three entries of each cell refer to 1, 2 and 4 wealth groups. An asterisk marks a tick that is not robust, because in at least one check of Section 5 an experiment fails or the smallest margin is below 0.10.

| Observed responses | Two-state example | Seven states, low wealth | Seven states, high wealth |
|---|---|---|---|
| **Changes in income risk only** | | | |
| Aggregate consumption | ✗ ✗ ✗ | ✗ ✗ ✗ | ✗ ✗ ✗ |
| Aggregate consumption and consumption of the poorer half | ✓ ✓ ✗ | ✓ ✓ ✗ | ✓ ✓ ✗ |
| **Changes in income risk and in expected income** | | | |
| Aggregate consumption and consumption of the poorer half | ✓\* group ✗ | ✗ ✗ ✗ | ✓\* ✗ ✗ |
| The same, with expected income and its standard deviation by group | ✓ ✓ ✓\* | ✓ ✓ ✓ | ✓ ✓ ✓\* |
| Aggregate consumption, with expected income and its standard deviation by group | ✓ ✓ ✓\* | ✓ ✓ ✓ | ✓ ✓ ✓\* |

**Aggregate consumption.** A rise in income risk common to all households lowers aggregate consumption, through precautionary saving, along a path that is almost proportional, with opposite sign, to the effect of the interest-rate cut. The uncentred correlation between the two responses is -0.72 in the two-state example and below -0.99 in the two seven-state parameterisations. With aggregate consumption alone, every change in income risk is attributed to monetary policy, in whichever wealth group it occurs.

**Consumption of the poorer half.** The interest-rate cut raises the consumption of the poorer half mainly in the first year, whereas a rise in income risk moves it along a different path, which in the low-wealth parameterisation is a decline that is still deepening after two years. The uncentred correlation between the two responses of this group is 0.15, 0.02 and -0.42 in the three parameterisations. With these data the method identifies the source and distinguishes up to two wealth groups.

**Risk and expected income.** When both types of change are allowed, consumption data do not tell them apart. In the two-state example the type of change is recovered only when the population is treated as a single group, and with two groups the wealth group is correct while the two types are confused. The seven-state parameterisations fail with one group under low wealth and beyond one group under high wealth. The two successes with one group are not robust, since in the two-state example an experiment fails in one check with the declared covariance matrix, and under high wealth the margin is 0.08 and experiments fail with the declared covariance matrix and with the shock to the discount factor.

**Income expectations.** When the data include expected income one year ahead and its standard deviation by group of households, source, wealth group and type of change are recovered with 1, 2 and 4 wealth groups in the three parameterisations and in all the checks. The expectations are grouped by wealth quartile in the two-state example and by quintile of earnings in the seven-state parameterisations, as the ECB Consumer Expectations Survey classifies households by income quintile. Since a quintile does not coincide with a set of earnings states, the households of a state that spans two quintiles are divided between them in proportion. With 4 groups and the declared covariance matrix the smallest margin is 0.08 in the two-state example and under high wealth and 0.29 under low wealth.

**Without the consumption of the poorer half.** Removing the consumption of the poorer half from these data does not change the outcome, and source, group and type of change are still recovered with 1, 2 and 4 wealth groups in the three parameterisations and in all the checks. The smallest margin is 0.46 with 1 and 2 groups, and with 4 groups and the declared covariance matrix it is 0.06 in the two-state example, 0.18 under low wealth and 0.03 under high wealth. The consumption of the poorer half therefore adds little to data that include expectations.

**Required precision.** The number of wealth groups must be fixed before looking at the data, since choosing it after seeing where the assignment succeeds would invalidate inference, as Leeb and Pötscher (2005) show. The code therefore computes from the model alone how precise the estimates must be. For a description with four wealth groups to be reliable, the consumption responses would need standard errors between 0.004 and 0.05 percent, an order of magnitude below those of typical estimates. Telling a change in risk from a change in expected income within a group, with 80 percent power at the 5 percent level, requires standard errors of the expectation data, in percent of expected income, between 0.4 and 2.3 in the two-state example and between 1.5 and 5.8 in the seven-state parameterisations. Without the consumption of the poorer half, the requirement under low wealth is between 2.0 and 2.5 with four groups and between 3.7 and 6.3 with two. For the average expectation of a group in a given month, this precision is within reach of a survey of several thousand households. The response of that average to a monetary policy shock is estimated on a sample that starts in 2020, and whether its standard error meets the requirement is an open question (Section 7).

## 7. Next step

The method is to be applied to the euro area with three sets of estimates.

- monetary policy shocks from the Euro Area Monetary Policy Event-Study Database of Altavilla, Brugnolini, Gürkaynak, Motto and Ragusa (2019),
- the response of aggregate consumption to these shocks, estimated by local projections on quarterly national accounts together with its covariance matrix,
- the responses of expected income and of its standard deviation by income quintile, from the ECB Consumer Expectations Survey. Every month since April 2020 the survey asks households how they expect their net income to change over the next 12 months, as a point forecast and as a probability distribution, and classifies them by income quintile.

The consumption of the poorer half of households is not among these observations, which are those of the last row of the table of Section 6, because no public source measures consumption by wealth in the euro area at quarterly frequency. The public files of the survey report the change in household spending by income quintile, and their only monthly information on wealth is an indicator of sufficient liquidity.

The first application distinguishes one or two wealth groups, for which the margins of Section 6 are wide and the linear approximation is accurate. The number of groups and the number $`N`$ of household histories are fixed before the estimates are examined. The number of groups is chosen by comparing the standard errors of the estimates with the precision requirements of Section 6. Since the survey starts in 2020, the responses of expectations rest on a short sample, and their standard errors are the main unknown of the application. The value of $`N`$ is the size of the sample of households on which the income process of the reference model can be tested, and the results are reported for a range of values, because with a given covariance matrix of the responses they depend on it (Section 8).

The reference model is the low-wealth seven-state parameterisation, which matches the liquid wealth of households and the share of households without liquid wealth. It is also the parameterisation in which the linear approximation is least accurate. The error is concentrated in the consumption of the poorer half, which the first application does not use, and in the changes of expected income of the poorer groups when four groups are distinguished (Section 8). With one or two groups, the responses used in the application are approximated with errors of at most 4 percent for expectations and 12 percent for aggregate consumption. As a check, the application will be repeated with the high-wealth parameterisation as reference, in which the errors are below 6 percent.

A description with four wealth groups, a second reference model with a looser borrowing limit and an income process estimated on EU-SILC data are left for a later stage.

## 8. Limits

- The experiments carry no estimation error, and the precision requirements of Section 6 are the only link to the sampling properties of actual estimates. The requirement on the consumption responses is computed for changes in income risk only and without expectation data. Its extension to the data of the first application is the first task of the next stage.
- With a given covariance matrix of the responses, the outcome depends on $`N`$. With the data of the last row of the table and the declared covariance matrix, the experiments with 1 and 2 wealth groups are successful in the two seven-state parameterisations for values of $`N`$ from 500 to 80,000, with a smallest margin that falls from 0.46 at 8,000 to 0.03 at 80,000. Those with 4 groups fail at 80,000 under low wealth and from 20,000 under high wealth. With the common variance, which is chosen again for each $`N`$, the outcome does not change.
- Uniqueness of the equilibrium is not proved. The code checks that the policy rule satisfies the Taylor principle and that the linearised system of equilibrium conditions is invertible at the reference, with smallest singular value 0.36, 0.34 and 0.69 in the three parameterisations.
- For each experiment of Section 5 the code reports the distance between the exact response and its linear approximation, relative to the size of the exact response, separately for aggregate consumption, the consumption of the poorer half, expected income and its standard deviation. With the monetary policy shock as aggregate disturbance, the error is below 6 percent in the high-wealth parameterisation and at most 21 percent in the two-state example, apart from one small response with 59 percent. In the low-wealth parameterisation it is concentrated in the consumption of the poorer half, with errors of 11 percent for the monetary policy shock, up to 37 percent for changes in income risk and up to 95 percent for changes in expected income. On expectations and on aggregate consumption, with 1 and 2 wealth groups the errors are at most 4 and 12 percent, the figures of Section 7. With 4 groups they are of the same size for changes in income risk, whereas a change in expected income in either of the two poorer groups is approximated poorly, with errors between 30 and 137 percent. The experiments are computed without approximation, so the table of Section 6 already reflects these errors.
- The model classifies households by current earnings, whereas the survey classifies them by the net income of the household in the previous year.
- The Phillips curve is not derived from a problem of price setters, and labour supply is inelastic.
- Households treat the transition probabilities of their wealth group as given and do not take into account that their saving may move them to another group.
- In the household problem of the two-state example the exact distance, computed with households re-optimising, differs from its quadratic approximation by about 6 percent at most for deviations of the probabilities up to 15 percent and by 18 percent at most for deviations up to 60 percent. The approximation also omits a factor between 1.08 and 1.11, almost uniform across earnings states, with respect to the exact second-order term in discrete time, and neglects the interaction between the two types of change.

## 9. Code

### Reproducing the results

```
pip install -r requirements.txt
make
python3 run_channels.py --closure hank
python3 run_channels.py --closure hank --expectations
HANK_CALIBRATION=liquid python3 run_channels.py --closure hank --expectations --groups income --sigma 0.2,0.3,0.9
HANK_CALIBRATION=liquid python3 run_channels.py --closure hank --expectations --groups income --sigma 0.2,0.3,0.9 --aggregate
HANK_CALIBRATION=liquid python3 run_channels.py --closure hank --expectations --groups income --sigma 0.2,0.3,0.9 --aggregate --households 80000
```

`make` runs the household block in isolation, the version with bond-market clearing and the HANK model in the two-state example, and takes a few minutes. The parameterisation is selected by the environment variable `HANK_CALIBRATION`, for instance `HANK_CALIBRATION=liquid python3 run_ge.py --closure hank`. The first two rows of the table of Section 6 are produced by `run_ge.py --closure hank`, the last three by `run_channels.py --closure hank`, with the options shown above for the expectations, with `--aggregate` to leave out the consumption of the poorer half and with `--households` to set $`N`$. The same script prints the error of the linear approximation for each block of responses.

### Names used in the code

| In this file | In the code |
|---|---|
| wealth group | region, `K` regions |
| change in income risk, $`\Delta^{\mathrm{risk}}`$ | `delta`, persistence, blocks `P1`, `P2`, ... |
| change in expected income growth, $`\Delta^{\mathrm{mean}}`$ | `drift`, blocks `D1`, `D2`, ... |
| monetary policy shock, discount factor shock | `eps`, `beta`, block `agg` |
| observed responses | moments |
| data without the consumption of the poorer half | `--aggregate` |
| expectations by quintile of earnings | `--groups income` |
| number $`N`$ of household histories | `NE`, `--households` |
| error of the linear approximation | `approximation` |
| experiment | injection |
| source indicated by the method | attribution |
| responses net of the effect of the real-rate path | `conditioned` |
| precision requirement | certificate, `required_se` |

### Files

| File | Content |
|---|---|
| `model.py` | household problem, steady state, wealth groups, alternative transition probabilities, perfect-foresight paths, expected income by group |
| `closure.py` | equilibrium with bond-market clearing, HANK equilibrium, linearised equilibrium systems |
| `entropic.py` | quadratic problem of Section 4, division of the minimised objective, detection error probability, precision requirement |
| `channels.py` | the two types of change and their experiments |
| `exact.py` | exact distance, at given decisions of households and with re-optimisation |
| `run_all.py`, `run_micro.py` | experiments on the household block in isolation, with aggregate consumption and with the consumption of the poorer half |
| `run_ge.py` | experiments with one type of change, with bond-market clearing or in the HANK model (`--closure hank`) |
| `run_channels.py` | experiments with two types of change |
| `run_exact.py`, `run_exact_reopt.py` | comparison between exact and approximate distance |
| `economics.py`, `figures.py`, `plotstyle.py` | descriptive statistics and figures of the two-state example |

### Earlier versions kept in the code

Two simpler versions of the economy preceded the HANK model and remain available. One solves the household problem at a given path of the real rate, without market clearing, and the other lets the real rate clear the bond market with exogenous income. In both, household income is the earnings state itself, so that a change of the income process also changes aggregate income. With aggregate consumption and the consumption of the poorer half, the version without market clearing recovers the source with 1, 2 and 4 wealth groups, and the version with bond-market clearing with 1 and 4 groups. `docs/nota_codice.pdf` describes the former in Italian.

## References

Acharya, S., and Dogra, K. (2020). Understanding HANK: Insights from a PRANK. *Econometrica*, 88(3), 1113-1158.

Altavilla, C., Brugnolini, L., Gürkaynak, R. S., Motto, R., and Ragusa, G. (2019). Measuring euro area monetary policy. *Journal of Monetary Economics*, 108, 162-179.

Auclert, A., Bardóczy, B., Rognlie, M., and Straub, L. (2021). Using the sequence-space Jacobian to solve and estimate heterogeneous-agent models. *Econometrica*, 89(5), 2375-2408.

Hansen, L. P., and Sargent, T. J. (2008). *Robustness*. Princeton University Press.

Kaplan, G., Moll, B., and Violante, G. L. (2018). Monetary policy according to HANK. *American Economic Review*, 108(3), 697-743.

Leeb, H., and Pötscher, B. M. (2005). Model selection and inference: Facts and fiction. *Econometric Theory*, 21(1), 21-59.

McKay, A., Nakamura, E., and Steinsson, J. (2016). The power of forward guidance revisited. *American Economic Review*, 106(10), 3133-3158.

Rouwenhorst, K. G. (1995). Asset pricing implications of equilibrium business cycle models. In T. F. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.

Werning, I. (2015). Incomplete markets and aggregate demand. NBER Working Paper 21448.

## Contact

Francesco Pinna, Department of Economics and Finance, LUISS Guido Carli, francesco.pinna@studenti.luiss.it
