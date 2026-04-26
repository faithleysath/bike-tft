# AGENTS.md

## Research Iteration Rules

- Treat every dataset, model, algorithm, and experiment change as part of the research record.
- Do not overwrite a previous experiment version in place when the change is methodologically meaningful.
- Create a new versioned directory or run name for each meaningful iteration, for example `nyc_top883_v2`, `agcrn_nyc_top883_v2_dep_arr_full_b64`, or `oracle_greedy_h12_top883_v2_cap200`.
- Keep a short README, manifest, result JSON, or run summary with each version so the thesis can reconstruct what changed, why it changed, and what result it produced.
- When comparing methods, preserve the older baseline outputs until the user explicitly says they are disposable.
- Prefer additive changes and clear version names over silent mutation of existing artifacts.
- If an experiment is only a smoke test, mark it clearly in the directory or run name.

