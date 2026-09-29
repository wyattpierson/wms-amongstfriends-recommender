# Golden sets — the answer key

A **golden set** is a human's opinion, for one taste profile, of what is a
**good** recommendation for *this listener* vs a **bad** one. The automated
leaderboard scores every model against the *same* list, so numbers are
directly comparable run-to-run and model-to-model.

```
eval/golden/
  jazzy_hiphop.json             ← one file per profile (same name as the fixture)
  wyatt.json
  ...
  labeled.jsonl                 ← auto-built by `make label` (per-rec verdicts, append-only)
  scores.jsonl                  ← auto-built by `make score` (1–5 response ratings, append-only)
  duels.jsonl                   ← auto-built by `make duel` / `make duels` (rankings → Elo, append-only)
```

The profile files are the ones **you edit by hand**. The `.jsonl` corpus files
are written by the computer — don't hand-format them; read them with
`make stats` or `make site`.

## The one rule that matters

**The fixture is the exam; the golden set is the answer key.** Every entry in
the golden set must be *inferable from the fixture's reviews*. If a verdict
isn't supported by what the listener actually wrote, the test is unfair and
the number means nothing. Before adding an entry, be able to point at the
review(s) that imply it. (This is why every entry carries a `reason`.)

## Entries

The most common form is still a bare string:

```json
"good": [ "MF DOOM", "Nas" ],
"bad":  [ "Post Malone" ]
```

The full form is an object:

```json
"good": [
  { "artist": "Nujabes",  "tier": "expected", "pair": "canon-vs-filler",
    "reason": "reviewed 5★ — the canon itself" },
  { "artist": "Robert Glasper", "album": "Black Radio", "tier": "novel",
    "reason": "cleanest out-of-lane bridge: jazz musicians + rap-adjacent" }
],
"bad": [
  { "artist": "Tomppabeats", "tier": "expected", "pair": "canon-vs-filler",
    "reason": "reviewed 2★: 'playlist filler, no soul in the digging'" }
]
```

| field    | required | notes |
|----------|----------|-------|
| `artist` | yes*     | artist name (*or write a bare string) |
| `album`  | no       | makes the match more specific |
| `tier`   | no       | `expected` (default) or `novel` — see below |
| `pair`   | no       | contrast-pair name, shared by a good-side and bad-side entry — see below |
| `reason` | no       | *(strongly preferred)* why, citing the review/taste pattern it comes from (`note` also works) |

All extra fields are optional — old-style files score exactly as before.

### Tiers: `expected` vs `novel`

- **`expected`** — in-lane, obvious from the reviews. The safe hit.
- **`novel`** — an *original move for this listener*: an adjacent scene, a
  cross-genre bridge, a deep cut. Novel is measured against the **listener**
  (new to *them*), not the model — the artist must still be one the model
  plausibly knows. Genuinely obscure picks wait until the model has earned
  confidence in its world knowledge.

The leaderboard reports **novel hits separately**, because "reaches for an
original-but-right move" is a different (rarer, more valuable) skill than
"echoes the lane back."

### Contrast pairs

A **pair** groups one good-side entry and one bad-side entry that are
near-identical *except* for the subtle difference this listener splits on.
This is what measures taste-inference instead of genre-echoing: a model that
just outputs "more of the same genre" *structurally cannot* score well,
because for every good-side pick there is an in-genre bad-side twin.

```json
"good": [ { "artist": "Nujabes",    "pair": "canon-vs-filler", "reason": "…" } ],
"bad":  [ { "artist": "Tomppabeats", "pair": "canon-vs-filler", "reason": "…" } ]
```

Scoring per pair (one unit, not two entries):

| engine recs… | pair score |
|---|---|
| the good side only | **+1** |
| neither side | **0** |
| the bad side (even *alongside* the good side) | **−1** |

Good pair candidates: same genre, different soul (canon vs filler); same
mood, different digging depth; same artist, different era/album (that old
Miles Davis trick). Write the `reason` on **both sides** — the "why does this
split?" is the whole point.

## How the score works

For each rec the model produces we normalize names (lowercase, whitespace
collapsed, `&`→`and`) and check:

1. Is **(artist, album)** on the good list? → good hit. On the bad list? → bad hit.
2. Otherwise is **artist** on the good list? → good hit. On the bad list? → bad hit.

An artist on *both* lists is ambiguous and only counts when the specific
album also matches:

```json
"good": [ { "artist": "Miles Davis", "album": "Kind of Blue" } ],
"bad":  [ { "artist": "Miles Davis", "album": "Tutu" } ]
```

→ *Kind of Blue* = good, *Tutu* = bad, any other Miles album = neutral.

Three numbers come out of one golden set:

- **`golden`** — base score: `(good_hits − bad_hits) ÷ n_recs`, clamped to [−1, 1]
- **`pairs`** — sum of contrast-pair scores (each −1/0/+1); leaderboard shows `points/total`
- **`novel`** — how many `tier:"novel"` good entries got hit (`hits/total`)

## Tips for a useful set

- **Start small, grow it.** ~12–15 entries per focused profile is plenty.
  Grow it from real use: when the same artist keeps coming up in
  `make label` / `make duel` sessions, promote it into the set. When taste
  changes, *edit* the entry.
- **Build the bad side from real engine output.** Run the engine a few times
  on the fixture (`make fast`), collect what it actually proposes, grade it,
  then add the tempting-wrongs as bad entries. Guarantees the set can
  actually discriminate.
- **Good picks should be plausible-but-not-obvious.** The 3 most famous
  artists in the genre make every model "pass" trivially.
- **Bad picks should be the near-misses** a lazy genre-centroid model would
  reach for, not random artists.
- **These are opinions about *this listener's* taste**, not your own
  ranking of favorites — the goal is "would *this profile* enjoy it."

## Using the set

```bash
make eval      # golden score + pairs + novel appear as leaderboard columns
make fast      # quick per-profile check while tuning prompts
make site      # static site shows the sets (and your live judging data)
```

`make label` uses the fixture to generate recs and lets you label them live;
the verdicts append to `labeled.jsonl` and — over time — crystallize into
this file.

## Starting from scratch

Copy an existing file, change `profile` (must match the fixture file's
`profile` field), fill in good/bad with `reason`s. `description` and
`annotated_by`/`annotated_at` are display-only. Coming from a **real user**?
`make fixture FB_UID=<uid> NAME=<profile>` fetches their reviews into
`fixtures/<name>.json` and scaffolds an empty golden seed here. An empty seed
scores `+0.000` for every model (no discrimination yet) — grow it.
