# Changelog

## [Unreleased]
### Added
- Expanded README with detailed usage instructions and CI badge.
- Milestone 7: Gradio UI scaffold (`rag_ed.ui.app`) with retrieval mode controls (vector/graph/fused), edge weighting & fusion parameter inputs, and diagnostics API (per-document vector vs graph score components).
 
### Future Implementation (Planned)
- Migrate vector store imports to `langchain_community.vectorstores` (remove FAISS/Chroma deprecation warnings).
- Add Canvas `module_order` edges once metadata is available.
- Introduce CLI flag `--graph-allowed-kinds` to expose edge kind filtering.
- Implement optional edge weighting (prioritize `same_stem`, `thread_reply`).
- Echo mode UX improvements (`ECHO:` prefix, optional context display).
- Multi-retriever fusion (graph + vector) with deterministic ranking heuristic.
- Structured edge statistics export (JSON) and zero-warning CI target.
 - Parse real Canvas `imsmanifest.xml` to populate `module_id` / `module_index` metadata.
 - Build a combined (Canvas + Piazza) unified graph for `--agent-type graph` instead of placeholder.
 - Complete vectorstore import migration (FAISS/Chroma) to silence remaining LangChain deprecations.
 - Document new `--graph-allowed-kinds` flag usage in README.
 - Milestone 8 (Planned): Real combined graph population, module metadata parsing, diagnostics enrichment (edge provenance, normalized similarity), UI/CLI smoke tests, zero deprecation warnings.

## [0.1.1] - 2025-08-26
### Removed
- Dropped PowerPoint (.ppt/.pptx) support to simplify dependencies.

### Added
- CI now runs ruff, black, mypy, pytest, and pip-audit.

