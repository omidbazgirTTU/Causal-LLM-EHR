# Oracle RWD Causal + LLM Robustness Study

## Goal
Quantify how LLM-induced measurement error affects causal inference in healthcare data.

## Core Idea
X* = true variable
X̂_m = LLM output

Bias(HR_m) = HR(X̂_m) − HR(X*)

## Models
- Model A
- Model B
- Qwen

## Pipeline
Raw → Cohort → Gold → LLM → Perturb → Causal → Compare
