# RAG-ed Troubleshooting & Change Log

This document provides a comprehensive overview of all problems encountered, errors faced, troubleshooting steps, and code changes made during the development and refactoring of the RAG-ed1 project. It is intended to serve as a reference for future maintainers and contributors.

---

## 1. Loader Refactor & Resource Management

### Problem
- Loader modules (`canvas.py`, `piazza.py`) returned file paths from temporary directories that were deleted after processing, causing file access errors in downstream code and tests.

### Troubleshooting & Solution
- Refactored loader logic to use a shared zip extraction utility (`extract_zip_to_temp`) with a callback pattern, ensuring all file processing occurs within the temp directory context.
- Updated tests to verify file existence inside the context and confirm cleanup after context exit.
- Used Python's `TemporaryDirectory()` for robust resource management.

### Challenges
- Ensuring all file operations happen before tempdir cleanup.
- Updating all loader and test code to use the new callback API.

### Resolution Steps
1. Created a shared utility for zip extraction with callback-based processing.
2. Refactored loader modules to use the utility.
3. Updated tests to check file existence before and after tempdir cleanup.
4. Validated with `pytest`.

---

## 2. CLI Agent Extension & Testing

### Problem
- CLI needed to support multiple agent types and robust testing, including mocking agent logic and subprocess-based CLI invocation.

### Troubleshooting & Solution
- Refactored CLI test to use parameterized agent types and monkeypatching for agent logic.
- Ensured CLI tests run in test mode with dummy data and mocked outputs.

### Challenges
- Mocking agent logic for all agent types.
- Ensuring CLI tests are isolated and reproducible.

### Resolution Steps
1. Parameterized CLI tests for agent types.
2. Used monkeypatching to mock agent logic and outputs.
3. Validated CLI via subprocess with dummy files and environment variables.
4. Confirmed expected output in test assertions.

---

## 3. Error Handling in Graph Modules

### Problem
- Missing artifact IDs in graph-related modules (`course.py`, `graph.py`) caused unhandled exceptions.

### Troubleshooting & Solution
- Added explicit `KeyError` handling with clear error messages for missing artifact IDs.
- Created negative-path tests to verify error handling.

### Challenges
- Ensuring all error paths are covered in tests.
- Providing clear, actionable error messages.

### Resolution Steps
1. Updated graph modules to raise `KeyError` for missing IDs.
2. Added tests to verify error handling and messaging.
3. Validated with `pytest`.

---

## 4. Docstring & Type Hint Refactor

### Problem
- Loader and retriever modules lacked clear docstrings and type hints, making usage and maintenance difficult.

### Troubleshooting & Solution
- Refactored docstrings to follow NumPy style, with detailed parameter and example sections.
- Added type hints to all relevant methods and constructors.

### Challenges
- Ensuring docstrings are accurate and comprehensive.
- Updating type hints without breaking existing logic.

### Resolution Steps
1. Updated docstrings in loader and retriever modules.
2. Added type hints to constructors and methods.
3. Validated with `mypy` and `ruff`.

---

## 5. Self-Querying Retriever Agent Integration & Testing

### Problem
- Needed to integrate `VectorStoreRetriever` with `vector_store_type="in_memory"` in the self-querying agent, and ensure robust agent creation and testing.
- Tests failed due to missing required arguments, real API calls, and tool interface mismatches.

### Troubleshooting & Solution
- Updated agent to use correct retriever instantiation and type hints.
- Created a dedicated test for agent creation, using monkeypatching to mock both the retriever tool and the model.
- Ensured dummy tool subclasses `Tool` and sets all required attributes.
- Updated test to robustly access the tool regardless of container type.

### Challenges
- Avoiding real API calls in tests (mocking model).
- Ensuring dummy tool matches required interface (`Tool` attributes).
- Handling different container types for `agent.tools`.

### Resolution Steps
1. Refactored agent to use correct retriever and type hints.
2. Created a dummy tool subclassing `Tool` with all required attributes.
3. Mocked model and retriever tool in tests.
4. Updated test to access tool regardless of container type.
5. Validated with `pytest`, `black`, `ruff`, and `mypy`.

---

## 6. General Troubleshooting & Validation

### Common Errors Encountered
- Type errors due to missing or incorrect type hints.
- Linter errors (unused imports, style issues).
- Test failures due to resource cleanup, API calls, or interface mismatches.

### General Steps Taken
1. Used `black` for code formatting.
2. Used `ruff` for linting and style checks.
3. Used `mypy` for type checking.
4. Used `pytest` for test validation after every major change.
5. Inspected and updated code/tests based on error messages and stack traces.

---

## Summary of Challenges & Solutions
- **Resource Management:** Solved by callback-based tempdir utility and context-aware tests.
- **Mocking & Isolation:** Solved by monkeypatching agent logic and models in tests.
- **Interface Compliance:** Solved by subclassing and setting required attributes for dummy tools.
- **Error Handling:** Solved by explicit exception raising and negative-path tests.
- **Documentation & Type Safety:** Solved by refactoring docstrings and adding type hints.

---

## Final Validation
- All tests pass except for one known unrelated type error in `vanilla_rag.py`.
- Code is formatted, linted, and type-checked.
- All modules and tests are robust, maintainable, and well-documented.

---

## Recommendations for Future Work
- Address remaining type errors and deprecation warnings.
- Continue to use context managers and callback patterns for resource management.
- Maintain comprehensive tests and documentation for all new features.

---

**End of Troubleshooting & Change Log**

---

## 7. Graph Edge Semantics Enrichment (2025-09-23)

### Problem
- The graph under `src/rag_ed/graphs` linked documents only by a simple, implicit chronological sequencing within a source directory. Edges were untyped and lacked semantics, limiting retrieval strategies and explainability.
- Request: “Please develop this subdirectory further by making more meaningful edges in the graph between documents.”

### Goals
- Preserve existing behavior while making edges self-describing (typed).
- Add low-dependency, high-signal relationships beyond plain adjacency to better reflect real-world associations between course artifacts.
- Keep the work testable in small increments with fast, fixture-free tests.

### Changes
1) Edge attributes support
	 - Allow edges to carry attributes (e.g., `kind`).
	 - File: `src/rag_ed/graphs/course.py`
	 - Change: `CourseGraph.add_relationship(self, source_id, target_id, **attrs)` now stores attributes on the underlying `networkx.DiGraph` edges.

2) Explicit chronological edges
	 - Tag existing per-directory, timestamp-ordered edges with `kind="chronological"` for clarity.
	 - File: `src/rag_ed/graphs/generation.py`
	 - Change: When linking documents by increasing `metadata["timestamp"]` within the same parent directory, annotate the edge with `kind="chronological"`.

3) Co‑temporal edges (same day)
	 - Connect documents that share the same calendar date (`YYYY-MM-DD`) across directories; bidirectional to represent undirected co-occurrence.
	 - File: `src/rag_ed/graphs/generation.py`
	 - Changes:
		 - Introduced `_safe_date_str()` to extract a date prefix from timestamps without extra deps.
		 - Grouped nodes by date and added edges `u->v` and `v->u` with `kind="co_temporal"`, only if an edge doesn’t already exist.

4) Same‑stem edges (cross‑format linkage)
	 - Connect files that share the same filename stem (e.g., `lecture1.md` and `lecture1.pdf`), across directories; bidirectional to represent the same logical artifact.
	 - File: `src/rag_ed/graphs/generation.py`
	 - Change: Group by `Path(source).stem` and add edges in both directions with `kind="same_stem"`, avoiding attribute overwrite.

5) Public helper for direct tests
	 - Expose a thin wrapper to build a graph directly from an in‑memory list of `Document`s.
	 - Files:
		 - `src/rag_ed/graphs/generation.py`: added `graph_from_documents(documents, *, prefix)` delegating to the internal builder.
		 - `src/rag_ed/graphs/__init__.py`: exported `graph_from_documents`.

### Why These Changes
- **Typed edges**: make relationships explicit for retrieval tuning and debugging (e.g., prioritize `same_stem` over `chronological`).
- **Co‑temporal**: capture cross‑directory temporal co‑occurrence (announcements, slides, notes published the same day) to surface contextually related artifacts.
- **Same‑stem**: link cross‑format versions of the same content (PDF, markdown, HTML), improving navigation and consolidation during retrieval.
- **Public test helper**: enables fast, isolated unit tests without relying on heavy Canvas/Piazza fixtures.

### Files & Diffs (Summary)
- `src/rag_ed/graphs/course.py`
	- Add: `add_relationship(..., **attrs)` to persist edge attributes.
- `src/rag_ed/graphs/generation.py`
	- Tag existing edges: `kind="chronological"`.
	- Add: `_safe_date_str()` and co‑temporal grouping/edges (`kind="co_temporal"`).
	- Add: same‑stem grouping/edges (`kind="same_stem"`).
	- Add: `graph_from_documents()` public wrapper.
- `src/rag_ed/graphs/__init__.py`
	- Export `graph_from_documents`.
- `tests/test_graph_generation.py`
	- Add: `test_generated_edges_are_tagged_chronological`.
	- Add: `test_co_temporal_edges_bidirectional`.
	- Add: `test_same_stem_edges_bidirectional`.

### Validation
- Command: `PYTHONPATH=src pytest -k "graph" -q`
- Result: All graph tests pass (9 passed, 18 deselected). Existing Canvas/Piazza graph generation and retriever tests remain green—no regressions.
- Deprecation warnings observed are unrelated to the graph code.

### Impact on the Original Problem
- The graph now contains “more meaningful edges” by:
	- Annotating temporal sequencing (`chronological`).
	- Adding cross‑directory temporal co‑occurrence (`co_temporal`).
	- Linking cross‑format representations of the same artifact (`same_stem`).
- Downstream systems (retrievers, agents) can:
	- Filter or weight traversal by edge kind for better relevance.
	- Explain why a neighbor was surfaced (edge kind provenance).
	- Extend with additional edge builders without changing core APIs.

### Challenges
- Avoiding attribute overwrite and edge duplication while keeping logic simple.
- Maintaining backward compatibility with existing tests and retriever traversal.

### Resolution Steps (Concrete)
1. Extended `CourseGraph` to accept/stash edge attributes.
2. Tagged existing adjacency edges as `chronological`.
3. Implemented co‑temporal and same‑stem bidirectional edges with safe dedupe checks.
4. Exposed `graph_from_documents` for small, fast unit tests.
5. Added three focused tests to validate new semantics.

### Recommendations / Next Increments
- Piazza threads (when metadata available):
	- `thread_reply`: `parent_id -> reply_id`.
	- `same_thread`: connect posts within the same `thread_id` (hub‑and‑spoke with root post).
- Canvas module order (when metadata available):
	- `module_order`: within `module_id`, link by `module_order` field.
- Retriever policy:
	- Optional `allowed_kinds` or weights to prefer `same_stem`/`thread_reply` over plain `chronological`.

---

## 8. Echo (Pass-Through) Mode for Agents (2025-10-06)

### Problem
- End-to-end testing of the retrieval + chain assembly required either real LLM API calls (slow, costly, flaky in CI) or the existing `TEST_MODE=1` shortcut which *skipped retrieval entirely* and returned a hard-coded string.
- Needed a middle ground: exercise the full retrieval stack (vector store build, prompt assembly) while avoiding external dependencies and costs.

### Solution
- Added an **echo mode** that replaces the OpenAI LLM with an in-process stub (`EchoLLM`) returning the final prompt unchanged. Retrieval and prompt construction still run; only generation is substituted.

### Implementation Details
- File: `src/rag_ed/agents/vanilla_rag.py`
	- Added `EchoLLM` subclass of `langchain_core.language_models.llms.LLM` implementing `_call` to echo the input.
	- Extended `one_step_retrieval(query, *, canvas_path, piazza_path, echo=False)` to select between `OpenAI` and `EchoLLM`.
	- Added CLI flag `--echo` and environment variable fallback `ECHO_MODE=1` (CLI flag precedence) to enable echo mode.
	- Preserved existing `TEST_MODE=1` behavior (continues to short-circuit and print `dummy answer`).
- Test: `tests/test_agents_cli_echo.py` invokes the CLI with `--echo` (without `TEST_MODE`) and asserts the original query appears in stdout while monkeypatching the retriever to avoid heavy I/O.

### Usage
```bash
python -m src.rag_ed.agents.vanilla_rag "What is RAG?" --canvas sample.imscc --piazza sample.zip --echo
```
or
```bash
ECHO_MODE=1 python -m src.rag_ed.agents.vanilla_rag "What is RAG?" --canvas sample.imscc --piazza sample.zip
```

### Comparison of Modes
| Mode            | Retrieval Runs | LLM Call | Output Behavior             |
|-----------------|----------------|----------|-----------------------------|
| Normal          | Yes            | Real LLM | Model-generated answer      |
| Echo (`--echo`) | Yes            | Stub     | Echo of final prompt/query  |
| TEST_MODE=1     | No             | None     | Literal `dummy answer`      |

### Impact
- Faster feedback loops during development (no API latency).
- Deterministic output for debugging prompt templates and retrieval context assembly.
- CI can validate integration without network access or cost.

### Future Enhancements
- Add a `--echo-prefix` option (e.g., `ECHO:`) for clearer visual differentiation.
- Provide a dry-run flag to print injected context blocks separately for inspection.
- Integrate echo mode into forthcoming Gradio UI for offline demos.

### Validation
- Ran existing CLI tests: unchanged (still rely on `TEST_MODE`).
- New echo test passes: confirms query text appears in output.

---

## 9. Synthetic Demo Dataset & Smoke Test (2025-10-06)

### Problem
Manual experimentation and future feature validation (e.g., thread edges, module ordering) required a tiny, deterministic set of input archives. Previously, tests relied on in-memory `Document` fabrication or heavier real export fixtures, slowing iteration and obscuring regressions.

### Goals
- Provide a reproducible way to generate a minimal Canvas IMSCC and Piazza ZIP locally.
- Add an end-to-end smoke test verifying loader → graph integration without network/API calls.
- Keep artifacts small, dependency-light, and deterministic (fixed timestamps in generators).

### Implementation
1. Script: `scripts/gen_demo_data.py`
	 - Uses existing test utilities: `tests.imscc_utils.generate_imscc` and `tests.piazza_utils.generate_piazza_export`.
	 - Writes two files to a target directory (default `demo_data/`):
		 - `demo_course.imscc`
		 - `demo_piazza.zip`
	 - CLI options: `--out-dir`, `--canvas-name`, `--piazza-name`.
2. Smoke Test: `tests/test_demo_data_smoke.py`
	 - Generates both archives in a temp directory (no reliance on repo state).
	 - Builds graphs via `graph_from_canvas` & `graph_from_piazza`.
	 - Assertions:
		 - Node count ≥ 1 for each graph.
		 - All edge kinds in the implemented set: {`chronological`, `co_temporal`, `same_stem`}.
		 - (Conditional) If ≥ 2 nodes, allows presence of chronological edges (future tests can tighten semantics).
	 - Includes helper `_edge_kind_counts` for potential future diagnostics.

### Files Added
- `scripts/gen_demo_data.py`: Developer utility to generate demo archives.
- `tests/test_demo_data_smoke.py`: End-to-end loader/graph validation.

### Rationale for Design Choices
- Reused test utilities to avoid duplicating archive generation logic.
- Kept assertions intentionally loose to remain stable as additional edge kinds are introduced later (e.g., `thread_reply`, `same_thread`).
- Avoided merging graphs in this test—kept per-source validation clear and focused.

### Validation
Command: `PYTHONPATH=src pytest tests/test_demo_data_smoke.py -q`
Result: Pass (1 passed). Two deprecation warnings (UTC naive `datetime.utcnow()`) noted in generators—benign and deferred.

### Impact
- Establishes a fast, deterministic baseline for upcoming semantic edge features.
- Simplifies manual reproduction: single script run produces ready-to-use artifacts.
- Provides early warning if loader or graph construction inadvertently regresses.

### Future Enhancements
- Suppress or modernize timestamp generation (timezone-aware `datetime.now(datetime.UTC)`).
- Add optional combined graph build & edge kind distribution logging.
- Extend smoke test once new edge kinds (thread/module) are implemented.

### Next Up
- Milestone 3: Introduce Piazza thread edges (synthetic metadata first) to expand conversational context semantics.

---

## 10. Piazza Thread Edges & Loader Integration (2025-10-06)

### Problem
Conversation structure in Piazza exports (threads, replies) was not represented in the course graph. This limited retrieval relevance and prevented conversational traversal strategies.

### Goals
- Surface conversational relationships from real Piazza exports into the graph.
- Keep synthetic unit tests for fast iteration, then ensure real loader output matches expectations.

### Implementation
1. Graph builder (`src/rag_ed/graphs/generation.py`):
	- Added detection for `post_id`, `thread_id`, and `parent_id` in `Document.metadata`.
	- Added `thread_reply` edges (parent -> reply). These overwrite coarse adjacency so parent/child semantics are explicit.
	- Added `same_thread` hub-and-spoke edges (thread root ↔ members) to provide thread-level context without O(n^2) connections.
	- Inserted safe precedence rules so `thread_reply` is preserved when `same_thread` is added.

2. Piazza loader (`src/rag_ed/loaders/piazza.py`):
	- Enhanced JSON parsing to expand `class_content_flat.json` into one `Document` per post when the JSON is an array.
	- Extracted post-level fields (`id` -> `post_id`, `thread_id`, `parent_id`) into `Document.metadata` so the graph builder can use them.
	- Robust handling for both dict-shaped page_content and JSON strings returned by the JSON loader.

3. Tests:
	- Unit test `tests/test_graph_thread_edges.py` validates synthetic Documents produce `thread_reply` and `same_thread` edges.
	- Integration test `tests/test_piazza_integration_threads.py` validates a generated Piazza ZIP produces post-level Documents and `thread_reply` edges in the graph.

### Validation
- Ran unit and integration tests:
  - `tests/test_graph_thread_edges.py`: passed
  - `tests/test_piazza_integration_threads.py`: passed
  - (Smoke tests and other graph tests previously passing remain green.)
  - Minor deprecation warnings from naive UTC timestamps in test generators were observed.

### Impact
- Real Piazza exports now produce thread-aware graphs. This enables retrieval strategies that can prefer conversational parents, show reply chains, or bias results to thread roots.

### Next Actions (short term)
1. Run the full test suite (`PYTHONPATH=src pytest -q`) and address any remaining regressions.
2. Add optional assertions in the integration test for `same_thread` hub edges to fully validate both edge kinds for real exports.
3. Implement `GraphRetriever.allowed_kinds` (Milestone 4) to let retrieval filter/weight by edge types (e.g., prefer `thread_reply` and `same_stem`).
4. Clean up deprecation warnings (timestamp handling) in test generators.

---

## 11. GraphRetriever Edge Kind Filtering (Milestone 4) (2025-10-06)

### Problem
After introducing typed graph edges (`chronological`, `co_temporal`, `same_stem`, `thread_reply`, `same_thread`), the retriever traversal treated all edge kinds equally. This limited experimentation with relevance strategies (e.g., focusing only on semantic edges like `same_stem` vs. temporal edges).

### Goal
Provide a minimally invasive way to constrain traversal to a subset of edge kinds without changing existing callers.

### Implementation
- File: `src/rag_ed/retrievers/graph.py`
  - Added optional constructor parameter `allowed_kinds: set[str] | None`.
  - During BFS traversal, an outgoing edge is considered only if either `allowed_kinds` is `None` (default legacy behavior) or the edge's `kind` attribute is in the provided set.
  - Preserves existing `max_depth` semantics and deduplication logic.
- Test: `tests/test_graph_retriever_allowed_kinds.py`
  - Constructs a tiny graph with a `chronological` edge and a `same_stem` edge from the same source node.
  - Asserts that the unfiltered retriever returns both neighbors, while a retriever restricted to `{"same_stem"}` returns only the same-stem target.

### Validation
Command: `PYTHONPATH=src pytest tests/test_graph_retriever_allowed_kinds.py -q` → Pass.
Full suite: 37 passed (after later deprecation fixes), confirming no regressions.

### Impact
- Enables future UI/CLI flags or adaptive policies to tune contextual expansion.
- Serves as a foundation for future weighting schemes (edge kind prioritization) without yet introducing complexity.

### Future Extensions
1. Weight-aware retrieval (scoring neighbors by edge kind before expanding).
2. Multi-phase traversal (e.g., first explore `same_stem`, then `thread_reply`, finally `chronological`).
3. CLI argument `--graph-allowed-kinds` to expose this filter at runtime.

---

## 12. Future Implementation Roadmap (Planned)

This section catalogs confirmed, higher-priority future changes not yet implemented. They are grouped by theme and list rationale, scope, and potential risks.

### A. Import & Dependency Modernization
1. LangChain Vectorstore Import Migration
	- Current Warnings: Deprecations for `FAISS` and `Chroma` imported from `langchain.vectorstores` in tests (monkeypatch context) and possibly source retriever code.
	- Plan: Replace with `from langchain_community.vectorstores import FAISS, Chroma`.
	- Scope: Update imports in any affected modules (`tests/test_retriever.py`, `src/rag_ed/retrievers/vectorstore.py` if present) plus adjust mocks if needed.
	- Risk: Minimal; ensure same constructor signatures; run full suite.
	- Acceptance: No LangChainDeprecationWarnings on full test run.

2. Timestamp Modernization (Completed in part)
	- Done: Replaced `datetime.utcnow()` in test utilities with timezone-aware versions.
	- Future: Audit any remaining naive UTC usage in source code (none currently flagged) and enforce via lint rule if added.

### B. Graph Semantics Expansion
1. Canvas Module Ordering Edges (`module_order`)
	- Rationale: Preserve curated pedagogical sequence beyond timestamp heuristics.
	- Input: Expected future metadata fields: `module_id`, `module_position` (or similar) from Canvas exports.
	- Implementation Sketch: For each module group, sort by `module_position` and add directed edges with `kind="module_order"`.
	- Tests: Synthetic docs with fake module metadata; ensure no cross-module edges.
	- Risk: Need reliable metadata extraction path—pending loader enhancements.

2. Edge Weighting / Scoring
	- Rationale: Some relationships are stronger signals (e.g., `same_stem` > `chronological`).
	- Plan: Introduce an optional weight mapping dict (`{kind: weight}`) consumed by retrieval for sorting expansion frontier.
	- Deferred: Keep BFS simple until concrete ranking needs arise.

3. Thread Root Summarization Node (Derived)
	- Idea: Auto-create synthetic hub node summarizing an entire thread (kind=`thread_root_synthetic`).
	- Use Case: Retrieval can inject a condensed version when space-limited.
	- Status: Concept only; would require summarization LLM or offline preprocessing.

### C. Retrieval & CLI Enhancements
1. CLI Exposure of Edge Kind Filtering
	- Add `--graph-allowed-kinds` accepting comma-separated kinds.
	- Validate against known kinds; fallback to all if empty.
	- Integration Test: Invoke CLI in echo mode with the flag and assert filtered context size changes.

2. Echo Mode UX Improvements
	- Prefix echo output with a marker (e.g., `ECHO:`) to differentiate from real answers.
	- Optional `--show-context` to print retrieved documents before answer.
	- Test: Validate marker presence and context block formatting.

3. Multi-Retriever Fusion (Graph + Vector)
	- Approach: Retrieve K docs from each, merge de-duplicating by source, then feed to chain.
	- Edge: Provide simple ranking heuristic (vector score vs. graph depth) as a testable function.
	- Risk: Must keep deterministic for tests without real embeddings (echo mode helpful).

### D. Quality & Observability
1. Structured Edge Statistics
	- Add helper to compute counts per edge kind and optionally serialize to JSON for diagnostics.
	- Test: Build graph from demo data and assert JSON keys present.

2. Warning Budget Zero
	- After import migration, aim for zero warnings in CI; enforce via `-W error` selectively (e.g., for known deprecations).

3. Lint Rule Additions
	- Consider ruff rules for disallowing `datetime.utcnow()` and enforcing timezone-aware usage.

### E. Stretch / Exploratory
1. Gradio or FastAPI UI
	- Minimal interface to upload archives, build graph, inspect edge kinds, and run echo retrieval queries.
2. Embedding Caching Layer
	- Cache vector embeddings keyed by content hash to speed repeated runs.
3. Knowledge Graph Export
	- Export to formats like GraphML / JSON-LD for external visualization.

### Tracking & Governance
- Each item should map to a GitHub issue with a label: `enhancement`, `graph`, `retrieval`, or `infra`.
- Milestone grouping:
  - Milestone 5: Module order & CLI filtering.
  - Milestone 6: Edge weighting & fusion retrieval.
  - Milestone 7: UI + observability utilities.
	- Milestone 8 (Planned Additions): Real Canvas module metadata parsing, combined graph build, vectorstore import migration, README flag documentation.

### Exit Criteria for This Roadmap Phase
1. No deprecation warnings on test run.
2. Module ordering edges implemented & tested.
3. GraphRetriever supports weighting (optional) and CLI filtering.
4. Zero naive UTC usage.
5. Echo UX improvements merged.

---

End of Future Implementation Roadmap.

---

## 13. Milestone 5: Module Ordering Edges & Graph CLI Filtering (2025-10-06)

### Problem
After enriching graph edge semantics and adding edge-kind filtering, explicit curricular sequencing (e.g., ordered Canvas module items) was still inferred only via timestamps. Additionally, the edge kind filtering capability wasn't yet accessible through the CLI for quick experimentation.

### Goals
1. Represent within-module ordering as first-class directed edges (`module_order`).
2. Expose edge kind filtering to users via a CLI flag for the graph agent.

### Implementation
1. Module Ordering Edges
	- File: `src/rag_ed/graphs/generation.py`
	- Logic: Collect nodes having both `module_id` and `module_index` metadata. Group by `module_id`, sort by integer `module_index`, and add directed edges between consecutive items with `kind="module_order"`.
	- Precedence: These edges intentionally overwrite coarse `chronological` edges between the same pair to reflect explicit pedagogical ordering.
	- Safety: Non-integer `module_index` values are ignored.

2. CLI Edge Kind Filtering
	- File: `src/rag_ed/agents/vanilla_rag.py`
	- Added flag: `--graph-allowed-kinds` (comma-separated). Parsed into a set passed as `allowed_kinds` to `GraphRetriever` when `--agent-type graph` is used.
	- Lazy Import: Retrieval class imported inside branch; patching path for tests simplified.

### Tests
1. `tests/test_graph_module_order_edges.py`
	- `test_module_order_edges_directed_and_overwrite`: Validates forward edges (1→2→3) are tagged `module_order` and reverse edges (if present due to other heuristics like co-temporal) are not mislabeled.
	- `test_module_order_ignores_non_integer`: Ensures non-integer indices do not produce `module_order` edges.
2. `tests/test_agents_cli_graph_allowed_kinds.py`
	- Confirms `--graph-allowed-kinds same_stem,thread_reply` passes `{"same_stem", "thread_reply"}` into the retriever.

### Validation
Selective test run:
```
PYTHONPATH=src pytest -q tests/test_graph_module_order_edges.py tests/test_agents_cli_graph_allowed_kinds.py
```
Result: All new tests passed.

### Impact
- Enables deterministic curricular sequencing independent of timestamp noise.
- Provides a quick, user-facing mechanism to experiment with traversal filters.

### Limitations / TODO
- Canvas loader does not yet extract real module metadata; current tests rely on synthetic metadata. Future enhancement: parse module structure from IMSCC manifest or module export files.
- Graph agent path still uses a placeholder (`course_graph=None`); needs integration to actually build and pass a combined graph.

### Next Steps
1. Integrate real module metadata extraction in `CanvasLoader` (parse `imsmanifest.xml`).
2. Provide combined graph construction for the graph agent (Canvas + Piazza union, de-dupe by source path).
3. Implement edge weighting for prioritizing `module_order` and semantic edges over temporal ones.
4. Add documentation snippet in README for `--graph-allowed-kinds` usage.

---

---

---

## 14. Milestone 6: Edge Weighting & Vector+Graph Fusion Retrieval (2025-10-06)

### Problem
While edge kind filtering (Milestone 4) and explicit module ordering (Milestone 5) improved semantic control, retrieval still treated all traversed edges uniformly and lacked a mechanism to jointly leverage dense vector similarity and graph structural signals. Tests required a deterministic, dependency-light fusion approach that could be validated without real embeddings.

### Goals
1. Allow callers to prioritize certain edge kinds (e.g., `same_stem` or `module_order`) over weaker temporal edges via numeric weights.
2. Provide a fused retriever combining vector-ranking and graph edge evidence into a single, stable ordering.
3. Expose fusion and weighting tunables through the CLI for experimentation.
4. Keep implementation deterministic and free of hashing pitfalls with LangChain `Document` objects.

### Implementation
1. Edge Weighting (GraphRetriever)
	- File: `src/rag_ed/retrievers/graph.py`
	- Added optional `edge_weights: dict[str, float]` argument.
	- Traversal collects neighbor candidates each BFS layer, sorts them by weight (desc) then node id for determinism, and records `(Document, weight)` pairs.
	- Removed internal `_last_scores` dict keyed by `Document` (unhashable) in favor of returning explicit score tuples.

2. Fusion Retriever
	- File: `src/rag_ed/retrievers/fusion.py`
	- Added `FusionConfig(alpha, beta, max_graph_depth, k)` and `CombinedRetriever`.
	- Scoring formula: `score = alpha * (1/(1+vector_rank)) + beta * edge_weight`.
	- Aggregates scores using `id(doc)` to avoid unhashable `Document` issues.
	- Stable ordering tie-breakers: (score desc, vector-priority flag, source path, id(doc)). Vector docs win ties against graph-only docs.

3. CLI Enhancements
	- File: `src/rag_ed/agents/vanilla_rag.py`
	- Added flags:
	  * `--retrieval-mode {vector,graph,fused}`
	  * `--graph-edge-weights kind:weight[,kind:weight]`
	  * `--fusion-alpha`, `--fusion-beta`, `--fusion-k`, `--fusion-max-graph-depth`
	- Provides a temporary placeholder `CourseGraph()` until unified graph construction is implemented (future milestone).

4. Determinism & Hashing Fixes
	- Initial attempt stored scores in dicts keyed by `Document`, causing `TypeError: unhashable type: 'Document'` in tests.
	- Refactored to use aggregation via `id(doc)` with explicit tuples; removed deprecated reliance on `invoke` path that required absent `tags` attribute in dummy retrievers.

### Tests
1. `tests/test_graph_retriever_weights.py`
	- Validates higher `same_stem` weight surfaces its neighbor before lower-weight kinds.
2. `tests/test_fusion_retriever.py`
	- `test_fusion_priority_with_weights`: Ensures ordering A (graph high weight) > Vec1 > B (graph low weight) > Vec2 based on combined scoring.
	- `test_fusion_tie_breaks_stable`: Confirms original vector order is preserved when only vector signals exist.
3. Existing graph filtering tests re-run to confirm no regressions.

### Validation
Selective run:
```
PYTHONPATH=src pytest -q tests/test_graph_retriever_weights.py tests/test_fusion_retriever.py
```
Result: All passed after hashing refactor.
Echo CLI test (`tests/test_agents_cli_echo.py::test_cli_vanilla_echo_mode`) still passes, confirming no regression from new CLI args.

### Impact
Provides a foundation for multi-signal relevance ranking without embedding score introspection. Users can quickly experiment with weighting schemes and alpha/beta trade-offs. Architecture supports future incorporation of real vector similarity scores once surfaced from the vector retriever.

### Limitations / TODO
* Fusion currently treats vector contribution solely via rank reciprocal, not true similarity magnitude.
* Placeholder empty graph in vanilla agent fused mode—needs real combined graph build utility.
* No README usage section yet (planned next step under documentation tasks).
* Edge weights only influence ordering, not traversal breadth (still bounded by `max_depth`). Potential future: weighted frontier prioritization (A* or best-first) for deeper graphs.

### Next Steps
1. Implement combined graph build from Canvas + Piazza loaders for CLI fused mode.
2. Surface actual similarity scores from vector store (normalize to [0,1]) to replace reciprocal rank heuristic.
3. Add README examples demonstrating tuning `--fusion-alpha` / `--fusion-beta`.
4. Provide optional JSON debug output of per-document score components.
5. Add integration test for fused mode inside CLI once combined graph builder is available.

---

## 15. Milestone 7: Gradio UI Scaffold & Retrieval Diagnostics (2025-10-06)

### Problem
Exploration of newly added retrieval capabilities (graph edge filtering, edge weighting, fusion parameters, echo mode) required a fast, interactive surface. The existing CLI imposed friction (repeated command invocation, limited visualization of component scores) and provided no structured diagnostics for iterative tuning.

### Goals
1. Provide a lightweight local UI to submit queries and adjust retrieval knobs (mode, allowed edge kinds, edge weights, fusion alpha/beta, depth/K) without restarting processes.
2. Surface deterministic diagnostics explaining how each context document was selected (vector vs. graph contribution components) to aid debugging and future ranking improvements.
3. Preserve offline operability via EchoLLM fallback when no API key is present.
4. Keep the implementation minimal (no heavy backend refactor) while leaving room for future graph visualization and provenance features.

### Implementation
1. Retrieval Factory (`src/rag_ed/retrievers/factory.py`)
	- Centralized construction of vector, graph, and fused retrievers based on a simple mode enum.
	- Added `parse_edge_weights` helper (kind:weight CSV) reused by CLI/UI.
2. Fusion Diagnostics (`src/rag_ed/retrievers/fusion.py`)
	- Extended `CombinedRetriever` with `retrieve_with_diagnostics` returning list of dicts: `{doc, vector_component, graph_component, final_score, source}`.
	- Ensured stable ordering matches production `get_relevant_documents` path.
3. Graph Build Placeholder (`src/rag_ed/graphs/build.py`)
	- Stub `build_combined_graph(canvas_docs, piazza_docs)` returning an empty `CourseGraph` (future Milestone 8 to populate) to keep UI wiring unblocked.
4. Gradio App (`src/rag_ed/ui/app.py`)
	- Blocks layout: sidebar controls (query text, retrieval mode selector, edge weights text input, numeric sliders for fusion parameters, depth/K), main panel (answer/echo output, expandable context list, raw diagnostics JSON preview/download button planned).
	- `run_query` orchestrates: build (or reuse placeholder) retriever, optionally call diagnostics path, format results.
	- Echo banner displayed when EchoLLM is in effect (planned improvement; base detection present).
5. Dependency Update
	- Adjusted `requirements.txt` to use `gradio>=5.8.0,<6.0` due to `smolagents` transitive constraint conflict with earlier `<5.0` pin.
6. CLI Compatibility
	- Left existing CLI behavior unchanged; future work will add an entry point (`rag-ed-ui`) and/or a `--ui` flag.

### Validation
Manual steps:
```
pip install -e .
python -m rag_ed.ui.app  # Confirmed local URL served, interactive controls render
```
Functional smoke via manual queries in echo mode; diagnostics JSON shows vector/graph component zeros where modes are exclusive (e.g., pure vector).

### Impact
Provides an experimentation cockpit for future feature work (Milestone 8 graph population, scoring refinements) and lowers barrier to tuning retrieval parameters. Establishes a channel to expose provenance data (scores, components) in a structured way.

### Limitations / Known Gaps
* Combined graph currently empty (no real semantic neighborhood context yet in graph/fused modes).
* No automated UI smoke test in `tests/` (planned).
* Diagnostics ordering not yet separately validated against display ordering (assumed consistent).
* No README or CHANGELOG summary entry yet (addressed in this milestone documentation process).

### Next Increments
1. Populate combined graph (Milestone 8) and re-run UI queries to validate non-empty graph context.
2. Add `tests/test_ui_smoke.py` calling `run_query` in echo mode with vector/fused to assert deterministic diagnostics shape.
3. Provide CLI/entry-point launcher and README usage section (install, run, echo fallback, tuning examples).
4. Optional: Add per-document expandable provenance panel (edge kinds traversed, weights) once combined graph is live.

---

## 16. Milestone 8 (Planned): Combined Graph Population, Module Metadata Parsing & Import Modernization (Target 2025-10-15)

### Motivation
With the UI scaffold and fusion framework in place, meaningful experimentation depends on a richly populated semantic graph aggregating Canvas (curricular structure) and Piazza (conversational threads). Additionally, outstanding technical debt (deprecated vectorstore imports, placeholder graph builder, missing module metadata extraction) risks accumulating friction and warnings.

### Objectives
1. Implement real `build_combined_graph` that ingests loader outputs, de-duplicates documents, and applies all edge enrichment passes (chronological, co_temporal, same_stem, thread_reply, same_thread, module_order).
2. Parse Canvas IMSCC (`imsmanifest.xml` and module-related resources) to attach `module_id`, `module_index` metadata enabling reliable `module_order` edges beyond synthetic tests.
3. Migrate all vector store imports to `langchain_community.vectorstores` to eliminate deprecation warnings (zero-warning goal for CI).
4. Enhance diagnostics: show per-document traversed edge kinds (first-hop provenance) and normalized vector similarity (replacing pure reciprocal rank heuristic when available).
5. Add UI and CLI smoke tests covering fused retrieval on the real combined graph.

### Planned Implementation Breakdown
1. Canvas Module Metadata Extraction
	- Parse `imsmanifest.xml` to map item identifiers to logical modules; derive ordering from manifest sequence.
	- Attach `module_id`, `module_index` to Documents at load time (update Canvas loader or a post-processing step).
2. Combined Graph Builder
	- Input: two `Sequence[Document]` (canvas_docs, piazza_docs).
	- Deduplicate by `(metadata.get('source') or metadata.get('path'))` + content hash fallback.
	- Run enrichment passes in precedence order (module_order overwrites chronological where overlapping).
3. Import Modernization
	- Replace all `langchain.vectorstores` imports with `langchain_community.vectorstores`.
	- Update tests/mocks accordingly; verify no `LangChainDeprecationWarning` remains.
4. Diagnostics Enhancements
	- Augment `retrieve_with_diagnostics` to include: `edge_kind` (if surfaced via graph), `rank_components` (dict with `vector_rr`, `edge_weight`, `final`), and optional `similarity` if available from vector store.
5. Testing
	- New tests: `test_combined_graph_non_empty_edges`, `test_ui_smoke_fused_mode`, `test_module_order_real_metadata` (using synthetic manifest fixture).

### Risks & Mitigations
* IMSCC schema variability → Mitigation: fallback graceful logging if manifest parsing fails; continue without module edges.
* Vector store API differences post-migration → Mitigation: add targeted unit test for retriever construction.
* Performance regression when building combined graph → Mitigation: hash-based dedupe and bounded traversal depth in tests.

### Acceptance Criteria
* Full test suite passes with zero deprecation warnings.
* UI fused mode surfaces at least one graph-derived neighbor (non-empty diagnostics with non-zero graph_component).
* README updated with UI & fused retrieval usage + troubleshooting.
* CHANGELOG entries (detailed + summary) reflect completion.

### Stretch (Optional)
* Graph visualization snippet (NetworkX → JSON → force layout) embedded in UI for selected document.
* Embedding cache keyed by content hash to accelerate repeated runs.

---


