# PolicyLens RAG

A citation-grounded RAG MVP for internal knowledge-base question answering.

## Overview

PolicyLens RAG is a practical Retrieval-Augmented Generation (RAG) project designed for internal document QA scenarios such as:

- leave policy lookup
- IT access procedure lookup
- SOP / policy retrieval
- evidence-grounded answers with citations

This project focuses on the core RAG pipeline rather than a flashy UI.

## Features

- document ingestion
- text cleaning
- page-level normalization
- token-aware chunking
- metadata design for provenance tracing
- local embedding support
- Chroma vector indexing
- retrieval with debug output
- citation generation
- basic abstention logic to reduce hallucination

## Current Architecture

1. Raw documents are placed under `data/raw/`
2. Documents are cleaned and transformed into page-level records
3. Pages are chunked into token-aware chunks
4. Chunks are embedded and indexed into Chroma
5. User queries are embedded and matched against indexed chunks
6. The API returns:
   - retrieved evidence
   - citations
   - fallback grounded answer
   - abstention flag when evidence is weak

## Project Structure

```text
app/
  api/
  core/
  schemas/
  services/
data/
  raw/
  processed/