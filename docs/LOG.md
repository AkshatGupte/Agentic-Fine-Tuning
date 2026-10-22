# LOG.md: Decisions and Run Notes

Add a short entry whenever you or the agent:
- change a setting from `docs/PROTOCOL.md`,
- make a manual choice (drop bad data, change the learning rate, pick a prompt),
- find a bug that affected scores,
- finish a Kaggle run.

These notes are important. Later, the training agent will need to make these same decisions automatically, so write down **what was decided and why**.

Keep entries short. Newest entry at the bottom.

---

## Entry format

```
### YYYY-MM-DD | Step N | short title
- What happened:
- Decision / change:
- Why:
- Effect on results (if known):
- Files changed / result files:
```

---

## Entries

### YYYY-MM-DD | Step 1 | Project started
- What happened: repo created, docs added.
- Decision / change: dev model chosen = `____` (1.5B or 3B). Spider stored at `____`.
- Why:
- Effect on results (if known): none yet.
- Files changed / result files: `AGENTS.md`, `docs/*`

---

## Open questions

- [ ] Confirm the Spider license is OK for this use.
- [ ] Final dev model: 1.5B or 3B?
- [ ] Frozen prompt: A or B? (decide at the end of Step 3)

---

## Run summary table

Fill this in as runs finish. Use `correct/total` next to the percentage.

| Date | Run name | Model | Prompt | Adapter | EX | Valid SQL | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
