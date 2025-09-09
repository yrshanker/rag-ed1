# RAG-ed1 Troubleshooting & Change Log

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
