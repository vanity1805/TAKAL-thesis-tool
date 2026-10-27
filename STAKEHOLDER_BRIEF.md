# TAKAL: Forecasting Retail Rice Prices in Metro Manila

**A plain-language project brief for non-technical readers**

*Prepared 27 October 2026. Figures come from the forecasting run of 26 October 2026, which used weekly data up to 27 April 2026.*

---

## 1. The short version

TAKAL (Time-Series Analysis & Knowledge Attribution for Local Rice) is a tool that does two things:

1. **It forecasts the weekly retail price of rice in the National Capital Region (NCR) for the next 12 weeks.**
2. **It shows which factors pushed each forecast up or down**, so that the result can be questioned and checked rather than simply trusted.

It uses six years of weekly records (330 weeks, January 2020 to April 2026) and a type of artificial intelligence that learns from patterns in past data. The results are viewed on an interactive web dashboard. Nobody needs to read or run any code to use it.

**The most important finding to know about:** adding economic indicators (fuel, exchange rate, imports and so on) did **not** make the forecasts measurably more accurate than using past rice prices alone. Section 6 explains this and what it means for how the tool should be used.

---

## 2. The problem we are addressing

Rice is the main staple food in the Philippines. When its retail price rises quickly, households that spend a large share of their income on food, and cannot simply eat less rice, have to cut back on other needs. Agencies, local governments and market planners usually learn about a price rise only after it has already happened.

Forecasting retail rice prices is hard for three reasons:

- **Many forces act at once.** Fuel costs affect transport, the peso–dollar exchange rate affects the cost of imports, harvests change local supply, and government stockpiles and import volumes change how much rice is available. Their combined effect is difficult to judge by eye.
- **Past prices help, but only so far.** Prices tend to move gradually, so last week's price is a good guide to next week's. But it says little about what happens when conditions change suddenly, such as the price spike of early 2024 or the unusually high diesel prices of early 2026.
- **Forecasts from "black-box" tools are hard to trust.** A tool that says "the price will rise" without saying why is difficult to use in public decision-making.

**TAKAL's aim** is a forecast that gives planners a few weeks of warning *and* explains itself, so that they can judge for themselves whether it is believable.

---

## 3. What TAKAL produces

| Output | What it tells you |
|---|---|
| **12-week price forecast** | The expected average retail price (₱ per kilogram) for each of the next 12 weeks. |
| **"Why" breakdown** | For each forecast week, how much each factor added to or subtracted from the price. |
| **Accuracy report** | How far off past forecasts were, measured on weeks the tool had never seen. |
| **Model comparison** | Whether the fuller model really beats a simple one, or whether the difference could be luck. |
| **Downloadable report** | A PDF or Excel summary of any of the above. |

---

## 4. How it works, without the jargon

### Step 1: Learn from history
TAKAL uses a kind of AI model called a **Stacked LSTM** (Long Short-Term Memory network). You can think of it as a forecaster that reads a short stretch of recent weeks, remembers which parts matter, and passes what it learned through three successive layers of analysis, each one refining the previous. At the end it gives its estimate for **all 12 coming weeks at once**.

How many recent weeks it looks at is chosen automatically by trying several options (2, 3, 4, 6, 8 and 12 weeks) and keeping the one that performed best on a validation set. In the current run, that was **2 weeks**.

### Step 2: Test it honestly
The 330 weeks are split in time order. The model learns from the **earlier part** (roughly the first 264 weeks) and is then tested on the **most recent 66 weeks** (27 January 2025 to 27 April 2026), which it never saw during learning. This is like setting an exam with questions the student has not practised. Future information is never allowed to leak into the learning stage.

### Step 3: Compare two versions
- **The fuller model** uses six pieces of information: recent rice prices plus fuel, exchange rate, rice stockpile, rice imports and palay (unmilled rice) production.
- **The simple baseline** uses recent rice prices only.

The baseline exists to answer a fair question: *do the extra indicators actually help?*

### Step 4: Explain each forecast
To explain forecasts, TAKAL uses a method called **SHAP**. The idea comes from fairly sharing credit among team members. Starting from a typical price (about ₱38 per kg for this model), SHAP works out how many pesos each factor added or subtracted to reach the forecast. The pieces always add up exactly to the forecast price.

> **Important:** SHAP describes how the *model* uses each factor. It does not prove that the factor *causes* prices to move.

### Step 5: Check whether differences are real
When two models score slightly differently, the difference may be luck. TAKAL applies a standard statistical check (the **Diebold–Mariano test**) to see whether one model is *reliably* better than the other.

---

## 5. The data behind it

The model learns from one spreadsheet, `THESIS-MASTER_DATASET.xlsx`, with **330 weekly rows** (one per Monday, from 6 January 2020 to 27 April 2026).

| Information | Unit | Range in the data | How often it changes |
|---|---|---|---|
| **Retail rice price** (the thing being forecast; the average of local and imported regular-milled rice) | ₱ per kg | 34.58 – 50.30 | Weekly |
| **Diesel price** | ₱ per liter | 27.84 – 153.70 | Weekly |
| **USD/PHP exchange rate** | ₱ per US$ | 47.73 – 60.98 | Weekly |
| **Rice imports** | metric tons | 5,049 – 607,760 | Updated about 73 times in 330 weeks* |
| **Rice stockpile** | metric tons | 187,733 – 3,158,540 | Updated about 75 times in 330 weeks* |
| **Palay (unmilled rice) production** | metric tons | 3,329,180 – 7,408,000 | Updated about 24 times in 330 weeks* |

\* These are published less often than weekly, so the same figure is repeated across weeks until a new one appears.

**Where the data originally came from:** the spreadsheet records *what* each column is but not *who published it*. The thesis team should complete the table below before this brief is shared outside the project.

| Information | Original publisher / publication | Access date |
|---|---|---|
| Retail rice price (local and imported) | *To be completed by the thesis team* | *To be completed* |
| Diesel price | *To be completed by the thesis team* | *To be completed* |
| USD/PHP exchange rate | *To be completed by the thesis team* | *To be completed* |
| Rice imports | *To be completed by the thesis team* | *To be completed* |
| Rice stockpile | *To be completed by the thesis team* | *To be completed* |
| Palay production | *To be completed by the thesis team* | *To be completed* |

**Data quality notes**
- The **last 9 weeks of rice imports** (March–April 2026) were not yet available and were filled in by repeating the last known figure (5,049 metric tons). This is a standard, documented approach, but it is an assumption.
- **Diesel prices in March–April 2026** (up to ₱153.70 per liter) are far above anything earlier in the data. They should be double-checked against the original source.
- Several indicators in the latest weeks (diesel, exchange rate, imports, stockpile) fall **outside the range the model learned from**. The model has never seen such conditions, so forecasts that depend on them are less dependable.

---

## 6. What the results say

### How accurate were the forecasts?
Measured on the 66 test weeks the model had not seen:

| Forecast distance | Typical miss: fuller model | Typical miss: simple baseline |
|---|---|---|
| 1 week ahead | ₱0.87 per kg (2.2%) | ₱0.73 per kg (1.8%) |
| 12 weeks ahead | ₱2.06 per kg (5.1%) | ₱2.31 per kg (5.7%) |
| **All 12 weeks combined** | **₱1.37 per kg (3.4%)** | **₱1.25 per kg (3.1%)** |

*"Typical miss" is the average gap between forecast and actual price.*

What this means in plain terms:
- Forecasts are **reasonably close in the near term** and **get less reliable the further ahead they look**.
- Twelve weeks out, both versions did worse, on a standard accuracy score, than simply guessing that prices would stay near the period's average. Long-range figures should be read as a rough direction, not a precise number.

### Did the extra indicators help?
**Not measurably.** Over all 12 weeks combined, the simple baseline was slightly more accurate, and the statistical test found **no reliable difference** between the two models (the result could easily be luck: p = 0.63; a difference is normally called reliable only below 0.05). This is a legitimate research result, not a failure of the tool. It means that, with the data available, recent price history carries most of the usable signal.

### The current 12-week forecast
The latest observed price was **₱43.74 per kg (week of 27 April 2026)**. The forecasts for the 12 weeks that follow (4 May – 20 July 2026) are:

| | First week (4 May) | Last week (20 July) | Change from latest price |
|---|---|---|---|
| Fuller model | ₱44.14 | ₱48.63 | about **+11%** |
| Simple baseline | ₱41.67 | ₱44.81 | about **+2%** |

Both versions end the period above the latest price, but they disagree about how far prices will rise. Because neither was shown to be reliably better, and the fuller model's inputs are partly outside what it learned from, **the forecast is best read as a rough direction, not as a precise number.**

### What drove the fuller model's forecast?
Averaged across the 12 forecast weeks, the factors pushing the fuller model's forecast (in ₱ per kg) were:

| Factor | Average push on the forecast |
|---|---|
| Recent rice prices | **+5.38** (up) |
| USD/PHP exchange rate | **+2.56** (up) |
| Rice import volume | **+1.93** (up) |
| Diesel price | **−1.22** (down) |
| Rice stockpile | **−0.40** (down) |
| Palay production | about 0 |

Diesel pushing the forecast *down* while diesel prices are at record highs is not what everyday reasoning would suggest. This is a good example of why the explanations show how the *model* behaves rather than how the *world* works, and why those unusually high inputs deserve a second look.

---

## 7. Using the dashboard

- **Overview** (default view): a simple summary of the forecast.
- **View Technical Details** (sidebar switch): opens four tabs: *Forecast & Overview*, *12-Week Forecast*, *Benchmarking / Metrics*, and *Explainability / SHAP*.
- **Forecast shown** (sidebar): switches everything between the fuller model and the simple baseline.
- **Export report**: downloads the selected sections as a PDF or Excel file.
- Every table has a short "Reading" note in plain language beneath it.

If the data cannot be loaded or the forecasting process fails, the dashboard now shows a clear message saying **what went wrong, where, and what to do next**, instead of a technical error screen.

---

## 8. Limits and responsible use

- **It is a decision aid, not a decision-maker.** Use it alongside expert judgement and current market information.
- **It is not live.** TAKAL learns from a prepared spreadsheet. It does not read real-time feeds, so a person must add new weeks of data and re-run the forecasting step to refresh it. The current forecast starts from the week of 27 April 2026, and its 12-week window has already passed. Adding the actual prices for those weeks and re-running the tool would show how well it really did.
- **Small dataset.** 330 weekly records is modest for this kind of AI model. Accuracy figures should be read with that in mind.
- **Unusual conditions are hard to forecast.** The model learns from the past, so sudden shocks it has never seen (such as the 2026 diesel spike) can mislead it.
- **Explanations are not proof of cause.** They describe what the model did, not why prices *really* moved.
- **Results can change.** Each time the model is re-run or new data is added, the numbers on this page can change.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **NCR** | National Capital Region (Metro Manila). |
| **Retail price** | The price consumers pay in markets, as opposed to farm-gate or wholesale prices. |
| **Palay** | Unmilled rice, as harvested from the field. |
| **Stacked LSTM** | A type of AI model that reads a sequence (here, weeks of data) and remembers what matters. "Stacked" means several layers working one after another. |
| **Multivariate / univariate** | Using many pieces of information / using one (past rice prices only). |
| **Forecast horizon** | How far ahead a forecast looks (here, 1 to 12 weeks). |
| **SHAP** | A method that splits a forecast into the amount each factor added or subtracted. |
| **Diebold–Mariano test** | A statistical check of whether one forecaster is reliably better than another or only lucky. |
| **Test weeks** | The 66 most recent weeks, kept hidden from the model during learning and used only to grade it. |
