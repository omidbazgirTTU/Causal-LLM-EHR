# Semantic LLM Policy

This file defines the default modeling rule for code and pipeline design in
this repository.

## Default Rule

- implement semantic LLM-based approaches by default
- do not introduce TF-IDF, BM25, bag-of-words, keyword scoring, sparse lexical
  retrieval, or similar heuristic text-matching pipelines as part of the main
  workflow
- when choosing between a lexical baseline and a semantic LLM approach, choose
  the semantic LLM approach

## Applies To

- note extraction
- reasoning pipelines
- evidence selection
- retrieval or context selection for prompts
- feature construction from free-text notes
- any new code that decides which text is relevant for downstream analysis

## Rationale

- this project is designed to study semantic LLM measurement behavior, not
  sparse lexical engineering
- lexical matching methods can weaken the scientific story when the goal is to
  evaluate modern LLM-based extraction and reasoning
- keeping the pipeline semantic-first makes the implementation easier to defend
  in a paper and easier to interpret as an LLM measurement study

## Do Not

- do not add TF-IDF vectorizers
- do not add BM25 ranking or keyword-count retrieval
- do not add bag-of-words or n-gram scoring as the main text-selection method
- do not fall back to lexical heuristics just because they are simpler to code

## If A Baseline Is Ever Needed

- only add a lexical baseline if the user explicitly asks for it
- keep it clearly separated from the primary semantic pipeline
- label it as a secondary baseline, not the default method
