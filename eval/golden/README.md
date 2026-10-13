# Golden sets — the answer key

A **golden set** is a human's opinion, for one taste profile, of what is a
**good** recommendation for *this listener* vs a **bad** one. The automated
leaderboard scores every model against the *same* list, so numbers are
directly comparable run-to-run and model-to-model.

```
eval/golden/
  jazzy_hiphop.json             ← one file per profile (same name as the fixture)
  full_spectrum.json            (replaces the old wyatt.json)
  underground_flo.json / alt_soul.json / club_kid.json
  ...
  labeled.jsonl                 ← per-rec good/bad/unsure verdicts (append-only). Written by
                                   `make label` (interactive) and by per-rec grading in `make score`,
                                   `make duel`, `make duels DUELS='… --grade-winner'`, and
                                   `make grade` (offline backfill from saved records)
  scores.jsonl                  ← auto-built by `make score` (1–5 response ratings, append-only)
  duels.jsonl                   ← auto-built by `make duel` / `make duels` (rankings → Elo, append-only)
  tag_proposals.jsonl           ← append-only: MusicBrainz suggestions from `make harvest-lookup`
                                   (sanitized; you copy the ones you trust into a profile's `tags`)
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

…and a fourth, response-level yardstick (next section).

## Structure: does the *response* mirror the fixture?

The good/bad score judges **items**. But a recommendation set is also a
*composition* — and the composition should **mirror the fixture**, not chase
generic diversity. A profile with four hiphop reviews deserves an
all-hiphop answer; a profile spanning electro-funk + neo-soul + art rap
deserves that range held. Diversity is only correct *because the fixture has
it* — so the checks below are per-fixture facts, and they obey the same one
rule as every other entry: **each constraint must be inferable from the
fixture's reviews** (write the `reason`).

Three optional top-level sections (all absent = the file scores exactly as
before). Piloted on `warm_groove` (genre-mirror) and `jazzy_hiphop`
(lane + era) — see those files for worked examples.

### `tags` — hand-annotated artist facts

```json
"tags": {
  "nujabes":  { "genres": { "coarse": "hiphop", "fine": "jazz hiphop" }, "era": "2000s" },
  "q-tip":    { "genres": "jazz hiphop" },
  "madlib":   { "genres": ["hiphop", "lo-fi"], "note": "also half of Madlib/Pharao" }
}
```

- `genres` accepts a level-dict (canonical), a bare string (→ `coarse`), or a
  list (`[coarse, fine]`). **Start coarse; add `fine` (and later `mid`)
  where it earns its keep.** No genre tree — levels are flat labels.
- `era` is a decade ("1990s"). At scoring time a rec's **Spotify
  `release_year` wins over the tag** (captured for free during verification);
  the tag is the fallback for unverified recs.
- **Untagged recs are neutral** — excluded from the check denominators, and
  the leaderboard shows tag coverage (`n% tagged ⚠️` below 60%). Coverage is
  your maintenance signal: grow `tags` from whatever keeps coming up.

### `constraints` — weighted checks on the response as a whole

```json
"constraints": [
  { "check": "lane_balance", "lane": "jazz hiphop", "area": "hiphop",
    "level": "fine", "value": 0.67, "weight": 2, "reason": "single-lane fixture…" },
  { "check": "min_distinct_decades", "value": 2, "weight": 1, "reason": "reviews span the 90s and 00s…" }
]
```

| check | params | meaning |
|---|---|---|
| `min_distinct_genres` | `level` (default `coarse`), `value` | ≥ N distinct genres at that level among tagged recs |
| `max_single_genre_share` | `level`, `value` | no genre above that fraction of tagged recs (anti-centroid-collapse) |
| `min_lane_share` | `lane`, `area`, `level` (default `fine`), `value` | mean lane credit ≥ value — graduated: **1.0** in the lane, **0.5** in the adjacent `area` (coarse), **0.0** elsewhere, untagged excluded |
| `lane_balance` | `lane`, `area`, `level` (default `fine`), `value` (default 0.5) | **per-rec net lane alignment, graded (partial credit).** Each TAGGED rec scores **+1** in the lane, **+0.5** in the adjacent `area`, **−1** tagged OUT of it; untagged recs are ignored. Mean ∈ [−1,+1] is mapped to [0,1] and earns *proportional* points — an off-lane rec actively subtracts, which a share check can't express. `value` is the pass line on that 0–1 scale (0.5 = net non-negative). Use this when "in the lane is good, out of the lane is bad" — prefer it over `min_lane_share` for that job. |
| `min_distinct_decades` | `value` | ≥ N distinct decades (Spotify year first, era tag second) |

**`weight`** (default 1) lets one constraint matter more than another —
structure = weighted points ÷ weighted total. Unknown `check` names are
skipped (not failed), so future checks can be added to files ahead of code.

**Untagged recs are neutral, and "unmeasurable" ≠ "failed."** Every check
runs only over recs it can actually judge (tagged genre, known decade). If a
check has *nothing* to measure — e.g. every rec is untagged — it is
`pass: null` (skipped, dropped from the denominator), **not** `false`. That
keeps an all-novel response from scoring 0.0 just because the tags haven't
caught up yet; instead you see a low `tag_coverage` (the harvest signal) and
a structure score computed over what *is* known. This is what lets the
engine outgrow the tag table without the score silently rotting.

### `justification_keywords` — does the *reason* carry taste signal?

```json
"justification_keywords": ["jazz", "boom-bap", "dig", "sample"]
```

`just` = fraction of recs whose `reason` mentions at least one keyword —
kept **informational, out of `quality` on purpose**: it measures whether the
justification is doing taste work ("it digs the same way the reviews dig")
without letting prose quality pollute the model comparison. A
`just_fixture_hits` count (reason names a fixture artist/album) is also
reported. Tiny lists — 3–6 words.

Leaderboard columns: per-profile **`structure`** (`points/total`) and
**`just`** (`hits/total`), plus a model-level **avg structure**. These sit
next to — never inside — `golden`/`quality`, so old runs stay comparable.

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

Per-rec grading is available in every judging flow, so you capture *what*
was good/bad, not just *which pool*:
- `make score` and `make duel` ask, right after the pool rating/verdict,
  whether to grade the individual recs (g=good / b=bad / u=unsure, optional
  note per rec).
- `make duels DUELS='agents --grade-winner'` grades the winner's recs after
  each batch duel.
- `make grade` (or `make grade LIMIT=3 MODEL=mistral`) is the offline
  backfill: it walks saved score/duel records — each one stored its full
  recs — and grades the ones you haven't judged yet. No model calls.

All verdicts land in `labeled.jsonl` tagged with their `source`. `make stats`
then shows per-source good_rate, a **pool-vs-item consistency check** (does
your 1–5 pool score agree with the recs you judged?), and **golden-set
promotion candidates** — artists with enough decisive labels who aren't in
the profile file yet (`make harvest` folds them in).

## Starting from scratch

**1. Get the fixture.** Coming from a **real user**?
`make fixture FB_UID=<uid> NAME=<profile>` fetches their reviews into
`fixtures/<name>.json` and scaffolds an empty golden seed here. Otherwise copy
an existing profile: change `profile` (must match the fixture file's
`profile` field) and re-derive everything below from *this* fixture.
`description` and `annotated_by`/`annotated_at` are display-only.

**2. Item level — `good` / `bad` (5–10 entries to start).** The canon this
listener clearly loves (good) and the tempting near-misses a lazy
genre-centroid would reach for (bad), each with a `reason` citing the review
that implies it. Add at least one **contrast pair** — that's what separates
taste-inference from genre-echoing. (A profile where item-matching "never
felt right" — e.g. the old `wyatt` — can legitimately start with empty lists and lean
on structure alone.)

**3. Response level — `tags` + `constraints` + `justification_keywords`.**
This is the part that *scales with the engine* (it judges shape, not
identity): read the fixture's reviews and write down the composition facts
they imply —
- What lanes/areas appear? → tag the ~5–10 artists you're confident about
  (start **coarse**; `fine` where it earns its keep).
- Single-lane fixture (all reviews in one lane)? → one `lane_balance` check
  (`lane` + `area`, weight 2). Multi-lane fixture? → `min_distinct_genres`
  (+ `max_single_genre_share` to prevent collapsing onto one lane), weight 1–2.
- Do the reviews span eras? → `min_distinct_decades`.
- 3–6 `justification_keywords` the fixture's reviews themselves use.
Every constraint gets a `reason` naming the reviews it was derived from —
same one rule as every other entry. Worked examples: `warm_groove` (multi-lane),
`jazzy_hiphop` (single-lane), `full_spectrum` (range-holding, multi-lane).

**4. Verify, then grow.** `make fast PROFILE=<profile>` — you should see
`structure` points, not `—`. Then run `make eval` a few times and let the
maintenance loop do its job: the `n% tagged ⚠️` column drops as the engine
recommends novel artists → `make harvest` lists them → `make harvest-lookup
ARTIST="…"` → you commit the ones you trust into `tags` (the tag table stays
small; **you are the commit step**). An empty seed scores `+0.000` for every
model (no discrimination yet) — grow it from real use.
