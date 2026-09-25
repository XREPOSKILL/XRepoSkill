# General Strategies for Repository-Level Issue Resolution

36 strategies distilled from successful SWE-bench agent trajectories across 11 Python repositories. Each strategy states when it applies (Trigger) and what to do (Action).

## 1. Reproduce the reported failure before touching source

**Trigger:** When starting a bug fix or implementing a reported missing behavior, before any source search or edit.

**Action:** Create and run a minimal standalone reproduction script, the existing failing test, or an isolated Python probe that exercises the exact failing input; install missing dependencies if needed and inspect the observed traceback/output to confirm current behavior. Keep that reproducer as the ground truth and re-run it after the patch to verify the fix.

## 2. Resolve Import and Dependency Issues Before Editing or Full Test Runs

**Trigger:** When imports, reproduction scripts, or test runners fail with missing-module/stale-package errors, or before full-suite validation after source edits.

**Action:** Keep the target package importable and current: reinstall in editable mode after source edits, install missing dependencies iteratively on ImportError/ModuleNotFoundError, and re-run a targeted import/reproduction until it succeeds. Fix import/dependency blockers before editing production code or running the full test suite; do not bypass a broken import by extracting code into isolated copies.

## 3. Use existing tests and grep all related call sites before/after patching

**Trigger:** When fixing a bug by modifying an existing function or rule and the expected behavior or full set of affected call sites is unclear.

**Action:** First run existing targeted tests or read relevant test cases to reproduce the expected behavior. Use grep/code search for related function names, error strings, constants, and sibling call sites, and apply the fix or guard consistently to all occurrences. After editing, run a quick smoke test such as an import or module invocation before the full test suite.

## 4. Probe Suspect Functions Interactively Before Editing Source

**Trigger:** When a suspect function or code path is identified and its behavior must be validated before source edits.

**Action:** Use an interactive Python shell or REPL to import and exercise the suspect code directly. Run multiple experiments probing edge cases, internal state, and return values before committing source edits.

## 5. Re-run the affected reproduction or test suite after every edit

**Trigger:** When you have a standalone reproduction script or affected test suite and are making iterative source or dependency edits.

**Action:** After each edit, immediately run the exact same reproduction script or affected test suite and compare its output with the pre-fix baseline. Do not batch multiple edits; verify each change immediately to catch regressions, partial improvements, or new errors as early as possible.

## 6. Inventory call sites and tests before modifying shared behavior

**Trigger:** When modifying a shared function, class-level attribute, incomplete implementation, or behavior that may have multiple call sites or lifecycle hooks.

**Action:** Search the codebase for the definition, all references/call sites, and existing tests before editing. Use that inventory to reveal symmetric or dependent paths and current coverage so the fix does not leave broken call paths.

## 7. Re-run a minimal reproducer after each edit before full tests

**Trigger:** When fixing a bug reported in an issue and you need to reproduce it or verify a patch.

**Action:** Before editing, write a small standalone reproduction script that exercises the reported feature using the exact issue input, and run it to confirm the current failure. Re-run that same script after every source edit until it passes consistently, and only then run the broader test suite to check for regressions.

## 8. Exercise Changed Code with Diverse Inputs and Repeat Affected Tests Before Full Suite

**Trigger:** when you add, change, or fix code whose behavior depends on input shape or values, or that may interact non-deterministically

**Action:** Run targeted snippets or the affected test module at least three times with diverse input variants, and inspect printed outputs directly before relying on the full suite. Then run the broader test suite to catch regressions.

## 9. Inventory all definitions and call sites of a shared symbol before editing

**Trigger:** When the fix involves a shared function, method, attribute, or pattern that may have multiple definitions, assignments, or call sites across the codebase.

**Action:** Search repository-wide for every definition and use form of the symbol or pattern (e.g. `def <name>`, `<name>(`, `.<name>(`, attribute assignments, and broader regex/factory patterns), read peer definitions with similar signatures or varying shapes, and list the affected sites before editing. Patch all affected call sites and related definitions consistently instead of changing only one location.

## 10. Apply Fixes to Sibling Identifiers or Arguments Symmetrically

**Trigger:** When you edit code to handle or convert one identifier or argument and similar sibling cases may exist nearby.

**Action:** Search the same function or nearby code for sibling identifiers or arguments that may need analogous treatment. Apply the same validation, conversion, or fix to them unless a comparable check already exists.

## 11. Reproduce the failure before searching or reading code

**Trigger:** When fixing a bug or starting from an issue, before searching code or reading source.

**Action:** Run the failing test/module or reproduction script first and read the actual traceback, so the failure is observable and the environment is proven functional. Do not search files or read source before reproducing the failure.

## 12. Resolve missing dependencies before inspecting or changing source

**Trigger:** When a reproduction attempt or import fails with missing-dependency errors such as ModuleNotFoundError or version incompatibilities.

**Action:** Do not proceed to source inspection or code changes; install the missing or incorrectly versioned dependencies, pinning exact versions if the error message specifies them, and rerun the minimal reproduction until the bug is actually observed.

## 13. Survey All Definitions and Call Sites Before Editing Behavior

**Trigger:** When modifying or debugging a function, class, method, signature, or attribute that may have multiple implementations, call sites, or hidden dependencies.

**Action:** Search the entire repository for the exact name and its bare-name, definition, call-site, and control-flow variants. Read those implementations and call sites to understand the expected contract and data flow, then apply the fix consistently across all affected paths.

## 14. Investigate the failure and existing conventions before editing

**Trigger:** When beginning a bug fix or regression and before changing source files.

**Action:** First reproduce or trace the failure to its exact token or module using a minimal script, traceback grep, or operator-dispatch check; then inspect existing code conventions, dispatch priorities, and paired or reflected methods before editing. Rerun the reproduction after each edit until the issue is resolved.

## 15. Run the test suite after every source edit

**Trigger:** After each source change or patch edit, before declaring the fix complete.

**Action:** Run the project test suite at the appropriate scope (targeted or full) immediately after each edit. Run it at least twice to guard against flaky failures, compare against a clean baseline for regression-sensitive changes, and reinstall first if the environment may be stale. Treat lightweight reproduction only as a prefilter, not a replacement.

## 16. Inventory peer definitions and symmetric call sites before editing

**Trigger:** When a bug fix touches a function or method that may have peers, callers, overrides, or delegated variants.

**Action:** Search for all related definitions and call sites (e.g., `def`/`class` declarations, delegated methods, symmetric code paths), compare signatures and patterns, and update every affected path consistently. Re-read the target function and verify with targeted tests or minimal snippets before finalizing.

## 17. Confirm a failing reproduction before source edits and re-run it after each edit

**Trigger:** When fixing a reported bug or before changing source code.

**Action:** Before editing, set up the environment if needed and create or run a minimal reproduction script or existing test to confirm the exact failure; do not edit source until it fails as reported. After each atomic source edit, re-run the exact failing command or test to validate the fix and catch regressions.

## 18. Read existing tests before editing source

**Trigger:** Before modifying source or tests, when the issue or affected code has associated tests or expected behavior to preserve.

**Action:** Read and run the relevant tests to identify expected inputs/outputs and edge cases before changing source. Ground the patch in those expectations, confirm you are editing the correct location, and update expected-failure markers if the fix intentionally changes previously expected failures.

## 19. Trace call sites and execution paths before changing behavior

**Trigger:** When modifying or changing a function's behavior.

**Action:** Search for the function name to inventory all call sites before modifying source, and map the execution path from entry point to target logic by reading the target function, identifying callees, searching for their definitions, and repeating. Include failure-path dependencies so no peer, upstream, or downstream location is left on old behavior.

## 20. Search related definitions and callers before and after code changes

**Trigger:** When modifying a function, class, or validation path whose behavior affects other definitions or test cases.

**Action:** Before editing, run definition-driven searches for relevant name patterns and read peer definitions, overloads, mixins, and callers. After editing, search the test suite for related class/model definitions or callers that exercise the changed path and verify no related tests break.

## 21. Inventory all related definitions before editing shared methods

**Trigger:** When a method or function has multiple implementations or sibling/peer definitions across classes/modules, or a change must preserve existing conventions and signatures.

**Action:** Search for all relevant `def` signatures and implementations, and read every related definition before editing to expose sibling code paths, shared conventions, and symmetric call sites.

## 22. Make repository source importable before running tests

**Trigger:** When a script or test must import the development version of the repository rather than an installed or system copy.

**Action:** Before importing the package, ensure the repository source takes precedence: use an editable install (`pip install -e .`) or prepend the repo root or compiled build directory to `sys.path` so modified modules and compiled extensions are loaded.

## 23. Grep existing attribute usage before modifying or adding one

**Trigger:** When you are about to modify an existing attribute or introduce a new timing/state attribute.

**Action:** Search the codebase for every dot-access read/write of the attribute and for analogous existing patterns, then replicate the established structure so no call site or convention is missed.

## 24. Inspect relevant definitions and source before editing

**Trigger:** When you are about to edit code to fix a bug or implement a feature, especially after reproducing an error.

**Action:** Before editing, search for and read the relevant function/class definitions and feature names to understand the call graph or pipeline order, and re-read the specific source file you plan to change. Do not edit until you have verified the current behavior and location.

## 25. Map definitions and call sites before first edit

**Trigger:** When starting to diagnose a bug or implement a fix, before making the first source-code edit.

**Action:** Run at least three cross-referencing searches for exact class/function/token names, callers, guard flags, and peer implementations; trace the issue-named operation or error message through its pipeline and read all related locations. Do not edit until alternative code paths and missed peer call sites are mapped.

## 26. Search existing implementations and reuse canonical patterns before editing

**Trigger:** When a patch touches a method or helper that already has overrides, peer entry points, or repeated validation/cross-validation calls elsewhere in the codebase.

**Action:** Before editing, search for all definitions and usages of the relevant method or helper, and read parent, sibling, and backend-specific implementations. Reuse the existing canonical guard/normalization/call pattern at the shared entry point or before loops, materializing/validating once so all overrides and repeated calls are covered consistently.

## 27. Reproduce exact failures with canonical project tooling before patching

**Trigger:** When starting a bug fix and before editing or writing standalone reproduction scripts, or when inspecting a failing expression whose Python truthiness may be misleading.

**Action:** Run the repository's official test runner with the exact documented selectors or test labels to confirm the failing case, and inspect the actual object or expression in that failing context using the project's preferred equality/truthiness predicates instead of generic Python truthiness. Verify the exact failing test name, selector, and runtime values before changing code.

## 28. Search every relevant symbol, peer call site, and definition before editing

**Trigger:** When a bug report, traceback, or feature names a specific symbol, keyword, option, function, or pattern, and before the first edit.

**Action:** Run whole-repository searches for the exact reported term and its sibling/complementary terms; then read every matching definition, call site, override, branch, and symmetric helper before choosing an edit site. If initial searches are too broad, narrow from file/directory globs down to keyword, function, and class-name searches before the first edit.

## 29. Inventory All Peer Call Sites and Sibling Components Before Changing Shared Code

**Trigger:** When modifying a shared or cross-cutting function, attribute, or API that has multiple call sites or sibling components.

**Action:** Search the entire codebase for every related occurrence and sibling definition before editing. Read each matching method body and plan to update all impacted locations together, adding tests for sibling components so the same logical fix is applied everywhere.

## 30. Search target definitions and surrounding call sites before editing

**Trigger:** When you are about to modify or debug a function, method, class, option, or shared helper and the issue names a specific symbol or code path.

**Action:** Run targeted definition searches (e.g. `grep -rn 'def <name>'`, class searches, or alternation searches for aliases/peer methods) and read the surrounding source; trace the call chain, callee signatures, call sites, overrides, and sibling/peer implementations. Inventory all relevant definitions and integration points, then edit only after the full usage landscape is clear.

## 31. Read relevant tests before editing source code

**Trigger:** When starting a fix for a failing issue, before editing source code.

**Action:** Locate and read the relevant or mirrored test file(s) that exercise the affected behavior, and reproduce or run the failing test when possible. Use the test assertions, expected values, edge cases, and any alternative spellings they reveal as the contract to guide the source edit and avoid regressions.

## 32. Use Combined-Pattern Grep to Inventory Relevant Sites

**Trigger:** When you need to locate all occurrences, peer implementations, validation sites, or test cases that involve multiple related terms before making a change.

**Action:** Run a single grep/search that combines the related patterns with alternation or a pipe (e.g. old_param|new_param or function|expected_header) to match files containing the target terms. Inspect the matched peers/tests and replicate or cover the existing behavior in every relevant method.

## 33. Map the full affected chain and patch all and only the necessary layers

**Trigger:** When an issue involves a setting, value, or compatibility error that propagates through multiple layers, call sites, or import chains.

**Action:** First trace the complete propagation path. If the change is an intended behavior or parameter update, update every downstream call site that must carry it; if the block is an import-compatibility error, apply the smallest patch to the file that unblocks the specific failing test, then run that test to get the true traceback before expanding.

## 34. Inventory related classes, callers, and guards before patching

**Trigger:** When a bug localizes to a class or function and the patch would modify or extend its behavior.

**Action:** Search for the containing class definition, its hierarchy and peer classes, all callers/usages, and existing defensive checks; read the relevant source, then fix every location sharing the same root cause rather than only the reproduction site.

## 35. Probe affected code with targeted reproducers before and after patching

**Trigger:** When a bug fix can leave untested branches or residual error/warning occurrences that the existing test suite may miss.

**Action:** Before editing, run a minimal reproducer or REPL script that exercises the affected function with representative and edge-case inputs. After editing, import and directly call the changed function and search for the same error or warning strings, handling every remaining branch and occurrence before treating the fix as complete.

## 36. Read full implementation, base, and peer classes before editing

**Trigger:** When fixing a bug, editing tests, or scripting a reproduction for behavior that involves inherited, peer, or downstream-call paths.

**Action:** Before patching, read the entire owning function/module, its base-class and sibling/peer implementations, and the associated test file; trace full call/propagation chains and re-read the target location immediately before editing.
