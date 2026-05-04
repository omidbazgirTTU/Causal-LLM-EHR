# ✅ Implementation Checklist: LLM + Causal Evaluation Pipeline (Multi-Model: + QWEN)

---

## 🔹 Phase 0 — Setup & Sanity Checks

- [ ] MIMIC-IV data downloaded and accessible  
- [ ] MIMIC-IV-Note dataset downloaded  
- [ ] Storage verified (sufficient disk space)  
- [ ] Data loaded into queryable format (DuckDB / Parquet recommended)  
- [ ] Can query:
  - patients  
  - admissions  
  - icustays  

---

## 🔹 Phase 1 — Cohort Filtering (DO THIS FIRST)

### ICU cohort
- [ ] Restrict to ICU stays (`icustays`)  
- [ ] Keep **first ICU stay per patient**  
- [ ] Filter to **adults (age ≥ 18)**  

### Candidate sepsis filter (loose)
- [ ] Keep patients with ANY:
  - antibiotic exposure  
  - lactate measurement  
  - blood culture  

### Time window
- [ ] Define:
  - `t0 = ICU admission`  
- [ ] Keep data in:
  - `[t0 - 24h, t0 + 48h]`

### Notes filtering
- [ ] Extract notes:
  - within time window  
  - physician + nursing notes  

### Output
- [ ] Create filtered dataset (~500–1000 patients initially)

**🚫 Do NOT run LLM before this step is complete**

---

## 🔹 Phase 2 — Gold Dataset (Human Labels)

### Sampling
- [ ] Select 200–500 patients  

### Annotation schema
- [ ] suspected_sepsis (yes/no)  
- [ ] suspicion_time (timestamp)  
- [ ] infection_source  
- [ ] evidence_span  

### Annotation
- [ ] Label notes (clinician or trained annotator)  
- [ ] Optional: double-label subset  

### Output
- [ ] Save as JSON or CSV  

---

## 🔹 Phase 3 — Multi-Model LLM Extraction

### Models
- [ ] Model A (e.g., GPT / OCI hosted)
- [ ] Model B (e.g., LLaMA / open model)
- [ ] Model C (QWEN)

### Setup
- [ ] Define strict JSON output schema  
- [ ] Use IDENTICAL prompt across all models  
- [ ] Enforce:
  - structured output only  
  - evidence included  
  - confidence score  

### Execution
- [ ] Run ALL models on SAME notes (gold dataset)

### Storage
- [ ] Save outputs separately:
  - `extracted_model_A.json`
  - `extracted_model_B.json`
  - `extracted_qwen.json`

---

## 🔹 Phase 4 — Extraction Evaluation

### Classification metrics
- [ ] Precision  
- [ ] Recall  
- [ ] F1 score  

### Time accuracy
- [ ] Mean error  
- [ ] Median error  
- [ ] % within 1 hour  
- [ ] % within 3 hours  

### Categorical match
- [ ] Compare infection_source  

### Optional
- [ ] Evaluate evidence span  

### Output
- [ ] Evaluation report per model  

### Comparison
- [ ] Compare Model A vs Model B vs Qwen  

---

## 🔹 Phase 5 — Build Analysis Datasets

### Dataset A (Gold)
- [ ] Use human-labeled suspicion_time  

### Dataset B (Model A)
- [ ] Use Model A outputs  

### Dataset C (Model B)
- [ ] Use Model B outputs  

### Dataset D (Qwen)
- [ ] Use Qwen outputs  

### Common variables
- [ ] treatment timing  
- [ ] outcome (mortality)  
- [ ] covariates  

---

## 🔹 Phase 6 — Causal Analysis

### Study setup
- [ ] Treatment: early vs delayed antibiotics  
- [ ] Outcome: 28-day mortality  
- [ ] Time zero: suspicion_time  

### Modeling
- [ ] Target trial emulation  
- [ ] IPTW weighting  
- [ ] Cox model  

### Output (per dataset)
- [ ] Hazard ratio (HR)  
- [ ] Confidence intervals  
- [ ] Survival curves  

---

## 🔹 Phase 7 — Core Comparison (KEY RESULT)

| Metric | Gold | Model A | Model B | Qwen |
|--------|------|--------|--------|------|
| Cohort size | ☐ | ☐ | ☐ | ☐ |
| Early treatment % | ☐ | ☐ | ☐ | ☐ |
| Hazard ratio | ☐ | ☐ | ☐ | ☐ |
| CI | ☐ | ☐ | ☐ | ☐ |

### Analysis
- [ ] Compare effect sizes  
- [ ] Check consistency of direction  
- [ ] Check overlap of confidence intervals  

---

## 🔹 Phase 8 — Sensitivity & Robustness Analysis

### Time perturbation
- [ ] Shift suspicion_time ±1–3 hours  
- [ ] Re-run causal model  

### Disagreement analysis
- [ ] Identify:
  - Model A ≠ Model B  
  - Model A ≠ Qwen  
  - Model B ≠ Qwen  

- [ ] Recompute causal effects on disagreement subset  

### Model stability
- [ ] Compute HR variance across models  
- [ ] Identify outlier model behavior  

---

## 🔹 Phase 9 — Final Outputs

### Tables
- [ ] Extraction performance (all models)  
- [ ] Cohort comparison  
- [ ] Causal estimates  

### Visualizations
- [ ] Survival curves (per model)  
- [ ] HR comparison plot  
- [ ] Sensitivity plots  

### Summary
- [ ] Robustness conclusions  
- [ ] Model differences (GPT vs LLaMA vs Qwen)  
- [ ] Failure modes  

---

## 🚫 Hard Rules

- [ ] Do NOT run LLM on full dataset  
- [ ] Do NOT include post-treatment notes  
- [ ] Do NOT tune extractor after causal results  
- [ ] Do NOT change prompts across models  
- [ ] Do NOT skip gold dataset  

---

## 🎯 Success Criteria

> Can we answer:  
> **“Are causal conclusions stable across different LLMs (including Qwen)?”**

---

## 🧠 Mental Model
