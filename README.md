# Minimum-action diagnostics of distributional misspecification in HANK models

Numerical laboratory for the M.Sc. thesis of Francesco Pinna (LUISS Guido Carli, Department of Economics and Finance).

## Research Question

A heterogeneous-agent New Keynesian model can reproduce the aggregate response of consumption to a monetary shock while misrepresenting how income risk is distributed across households, because errors of opposite sign in different regions of the wealth distribution offset in the aggregate. The diagnostic studied in the thesis asks which deformation of the household income process, at minimum informational cost, reconciles the model with a set of observed moments, and where in the wealth distribution that cost is spent.

## Role of this repository

The code runs a controlled experiment on a one-asset Aiyagari economy at quarterly frequency. A known misspecification of the income process is injected in a known region of the wealth distribution, the resulting consumption response is computed on the nonlinear perfect-foresight path, and the diagnostic is asked to attribute the gap. Because the truth is known by construction and the experiments carry no sampling noise, every attribution error reflects the geometry of the problem.

The deformation scales the probability of switching income state for households whose end-of-period wealth falls in a given region. Its cost is the second-order relative entropy of the deformed income process with respect to the baseline, which weights each region by the expected number of income transitions originating there. An aggregate channel, a misspecified path of the real rate, competes with the regional deformations.

## Economic interpretation of main findings

The aggregate consumption response makes the two tails of the wealth distribution visible and the middle almost invisible. In the bottom quartile, where virtually no household holds the high income state, more frequent income transitions work as an improvement in income prospects for households that consume 80 percent of a transfer within the quarter. In the top quartile, where 86 percent of households hold the high state, the same deformation works as a worsening of prospects for the households that own 57 percent of wealth. In the second quartile the two states are almost equally represented, the effects on expected income offset, and the deformation leaves almost no trace on aggregate consumption.

On aggregate moments alone, a uniform increase in income churn is almost the mirror image of a rate cut (cosine of about -0.86), so that a single-region diagnostic attributes the churn injection to the monetary channel with a large margin. Adding twelve quarters of consumption of the bottom half of the wealth distribution removes the confounding, because on that group the two mechanisms move in the same direction while on the aggregate they move in opposite directions. The granularity at which misspecification can be localized is therefore a joint property of the chosen regions and of the observed moments.

Attribution by raw cost shares or by pairwise contrasts is biased towards the blocks with the largest aggregate footprint, the monetary channel and the wealthiest region. Attribution by the fraction of each region's capacity that is actually used recovers every injected region, including a deformation whose detection error probability is 0.465.

## Limits

The laboratory is in partial equilibrium, since the real rate is exogenous and no asset market clears, so the Jacobian is the direct household-block sensitivity without the general-equilibrium feedback term. The symmetric deformation changes income persistence and, through the income-state composition of each region, expected income at the same time, so it does not yet separate a correction of propagation from a correction of risk. Both points define the work of the thesis.

## Reproducing the results

```
pip install -r requirements.txt
make          # runs run_all.py, run_micro.py, economics.py, figures.py
make docs     # compiles the notes in docs/ (requires a LaTeX installation)
```

The full pipeline runs in under a minute on an ordinary machine.

| File | Role |
|---|---|
| `model.py` | baseline economy, steady state, wealth regions and transition weights, income deformation, perfect-foresight transition path, Jacobian by direct anticipated perturbations |
| `entropic.py` | prior precision, penalized minimum-cost correction, exact cost decomposition, capacity, detection error probability, Imhof quadrature for the null distribution |
| `run_all.py` | controlled injections and attribution on the nested family of regions with aggregate moments only, numerical validations, sensitivity to the effective panel size |
| `run_micro.py` | the same analysis with bottom-half consumption moments added |
| `economics.py` | economic descriptors by wealth quartile, consumption responses, rate and churn confounding |
| `figures.py` | figures of the June numerical appendix |
| `plotstyle.py` | common figure style, deterministic PDF output |

`docs/nota_codice.pdf` describes the economic content of the results and the structure of the code (in Italian). `docs/appendice_numerica_toy.pdf` is the numerical appendix of June 2026, which documents the protocol in full.

## Contact

Francesco Pinna, Department of Economics and Finance, LUISS Guido Carli, francesco.pinna@studenti.luiss.it
