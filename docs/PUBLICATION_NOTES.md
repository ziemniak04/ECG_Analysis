# Publication scope and local-only files

The exact intended repository tree is enumerated in `PUBLICATION_MANIFEST.json`, including sizes and SHA256. The manifest itself is included separately to avoid a self-hash cycle. Source code, model definitions, configurations, automated tests, historical technical documents and aggregate metrics are included. No waveform arrays, per-record predictions, model weights, virtualenvs or credentials are included.

The initial local commit already tracked two PTB-XL metadata CSV files, a large download manifest and a pilot manifest. The publication procedure untracks those files and the old nested data .gitignore in a NEW commit, keeping every local file. The clinical database CSV contains demographics and ECG reports, so is excluded even though it comes from a public benchmark. Removing tracking does not erase old history. History is not rewritten. If excluded data exists in commits that are not already reachable from origin, the script stops rather than newly uploading that historical content.

Raw and processed datasets, checkpoint files, detailed patient-level manifests/results, all `results/` artifacts, caches and secrets remain local-only. Frozen split configuration uses public benchmark record/group identifiers, not actual personal identities. Only aggregate final metrics have been copied into `docs/experiments/`; no individual waveform, demographic row, upload or patient prediction is copied there.

To review without changing Git or accessing the network:

```bash
.venv/bin/python scripts/publish_verified_handoff.py
```

To publish from the owner's normal VS Code terminal AFTER checking the audit:

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python scripts/publish_verified_handoff.py --publish
```

The explicit `--publish` path verifies origin and current branch/HEAD, fetches remote refs, stops for new remote FEATURE commits/divergence; newer main commits are informational and remain untouched, and checks that excluded historical files would not be newly exposed. It stages only enumerated files, untracks excluded local files without deleting them, reviews all staged blob paths/sizes/secrets and diff checks, commits and pushes the current feature branch without force. It rechecks the remote tip to avoid racing teammate changes and verifies the resulting remote hash. It does not merge/rebase/delete branches, retrain, reevaluate or read ECG arrays. Network/authentication and writable .git are required. Failure stops; do not bypass by using force options or ignoring the failed audit.

If the remote feature branch has new commits, inspect with `git log --left-right --graph --oneline HEAD...origin/feature/ptbxl-1dcnn`. Resolve collaboration separately; the script will not do it automatically. Anna can fetch/pull the feature branch only AFTER a successful push is confirmed; the existence of these files is not evidence that GitHub was updated.

The known16 commits on origin/main are intentionally NOT merged/rebased. See FEATURE_BRANCH_PUBLICATION_PLAN.md for reviewed commits, exact/semantic overlap and checkpoint/RR incompatibilities. A verified existing publication commit is reused instead of duplicated; the only permitted push target is feature/ptbxl-1dcnn.
