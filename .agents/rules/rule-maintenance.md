# Rule Maintenance & Synchronization

## Keeping Rules & Skills Up-to-Date
- **Adding Functionality**: Whenever new modules, files, types, macros, or core workflows are introduced:
  - Update the Source Code Map in `AGENTS.md`.
  - Update relevant `.agents/rules/` and navigation maps in `.agents/skills/easyhybrid-dev/SKILL.md`.
  - Add corresponding unit tests in `test/` and update `test/runtests.jl`.
- **Modifying or Deleting Functionality**: When refactoring, deprecating, or deleting code:
  - Remove or update references in `AGENTS.md`, `.agents/rules/`, and `.agents/skills/`.
  - Update or remove obsolete tests in `test/`.
  - Ensure no orphaned imports, dead code, or outdated documentation remain.
- **Continuous Documentation**: Keep public API docstrings and user guides synchronized with changes at all times.
