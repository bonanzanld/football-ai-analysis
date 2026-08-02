# Ball detection and trajectory QA

## Current state

The ball tracker combines RF-DETR detections with temporal selection. It can:

- compensate for camera movement before comparing candidates;
- bridge short detection gaps with bounded prediction;
- interpolate plausible gaps retrospectively;
- retain a stationary ball between reliable endpoint detections;
- use nearby player footpoints as supporting evidence;
- use player proximity as a bounded tie-breaker between similarly credible
  initial ball candidates;
- confirm coherent weak motion after a strong player-contact frame, allowing a
  fast early kick to establish velocity without trusting one weak detection;
- run an adaptive high-detail crop around the last credible ball area, because
  a small ball loses substantial detector confidence when the full 4K frame is
  resized for inference;
- reacquire a weak, persistent candidate near active play after prolonged
  detector loss;
- start a new trajectory segment when that weak reacquisition is remote, so
  retrospective interpolation cannot draw a false path back to the old ball
  position.

The relevant implementation is in
`src/football_ai/detection/ball_tracking.py`. The focused regression suite is
in `tests/test_ball_tracking.py`.

## Reproduce the focused tests

Use the repository virtual environment:

```bash
.venv/bin/python -m pytest tests/test_ball_tracking.py -q
```

Current result on 2026-08-02: 102 focused tests passed. The complete project
suite has 408 passing tests.

## Generate a QA report

Analyze the first minute of a video:

```bash
MPLCONFIGDIR=/tmp/football-ai-mpl \
  .venv/bin/python tools/analyze_ball.py \
  --video TrainingClipXBOTGO_High.mp4 \
  --seconds 60 \
  --threshold 0.05
```

The command writes:

- `output/ball/TrainingClipXBOTGO_High_ball_qa.mp4`;
- `output/ball/TrainingClipXBOTGO_High_ball_tracking.json`.

The JSON distinguishes physical detections from `predicted`, `interpolated`,
and `stationary_hold` observations. It also assigns a `track_segment` number;
a confirmed switch from a weak wrong track to the active player cluster starts
a new segment, so consumers must not interpret that switch as physical ball
movement. Coverage must therefore never be treated as detector recall by
itself.

## Latest baseline

The 60-second `TrainingClipXBOTGO_High` run on 2026-08-01 produced:

| Source | Frames |
| --- | ---: |
| detected | 930 |
| predicted | 145 |
| interpolated | 201 |
| stationary hold | 163 |
| total visible | 1439 / 1798 (80.0%) |

Camera stabilization linked 1797 frame transitions and skipped none. The
first 30 seconds contain 821/899 visible frames (91.3%); the second half has
618/899 (68.7%). The run deliberately leaves uncertain ranges empty: a 7%
detection on a player's
head had previously created a false track segment at frame 426. Candidates
inside the head or torso are now rejected, while the real ball is reacquired
at a player's foot at frame 497. The preceding 82.9% run is therefore not a
valid improvement: its additional coverage included this false positive.

The minute run also exposed two remote false starts: a training cone at frame
1069 (23.3%) and sideline clutter at frame 1296 (17.3%). Requiring 30% for a
remote weak restart removes both false segments.

Targeted checks around frames 1652-1655 and 1777 contain long, dark player
shadows. The marker stays on the visible ball in those reviewed frames and
does not jump to a shadow. This is positive case evidence only, not a general
claim that shadow robustness is solved.

The pass sequence around 19:24:40 previously produced an 80-frame gap at
frames 1055-1134 even though RF-DETR detected the visible ball at 65-83%
confidence. The stationary-track and two-player restart gates, not the
detector, caused the loss. A strong candidate of at least 50% at one player's
foot may now restart only after three coherent frames. The gap is reduced to
frames 1055-1063, and 67/76 frames in the reviewed 1045-1120 window are now
visible. Visual checks confirm the marker follows the passes between players.

## How to evaluate a change

1. Run the focused tests.
2. Generate fresh QA with the command above.
3. Inspect the longest uncovered frame ranges in the JSON.
4. Inspect frames around every new reacquisition and large image-space jump.
5. Compare the marker with the active ball, not merely the nearest white
   circular object.
6. Re-run the user-provided amateur reference clips before accepting a tracker
   change. Do not use professional match footage for development or QA.

Do not optimize for coverage alone. A longer but incorrect trajectory is
worse than leaving uncertain frames empty.

## Ground-truth precision and recall

Trajectory coverage includes predictions and retrospective filling, so it
cannot measure detector recall. Export a small set of original video frames
for manual review instead:

```bash
.venv/bin/python tools/prepare_ball_ground_truth.py \
  --video brandevoortBRAB_vid2.mp4 \
  --start-frame 400 \
  --end-frame 500 \
  --step 5
```

The command creates `data/ball_ground_truth/brandevoortBRAB_vid2/annotations.json`
and the corresponding JPEG images. For every frame, set `visibility` to
`visible`, `occluded`, or `not_visible`. A visible or partly occluded active
ball needs an original-image `ball_box`. Keep `unreviewed` only while work is
in progress; those frames are excluded from metrics. The manifest explicitly
requires human review. Exported JPEGs are ignored by Git, while the compact
annotation JSON remains versionable.

Open the local review interface:

```bash
.venv/bin/python tools/review_ball_ground_truth.py \
  --annotations data/ball_ground_truth/brandevoortBRAB_vid2/annotations.json
```

Click the center of the active ball to place a 20-pixel box; `+` and `-` resize
it. Mouse dragging remains available when a custom box is easier. Use the
on-screen buttons to save the ball as visible, player-occluded, or not visible;
the next frame opens automatically. Keyboard shortcuts remain available, but
are not required. The cursor magnifier helps with small distant balls. Every
confirmation is saved immediately; `A` and `D` navigate between frames. Only
entries explicitly saved by this interface as `human_reviewed` participate in
the metrics.

Collect raw and person-filtered detector candidates on only the annotated
frames, then evaluate both stages:

```bash
MPLCONFIGDIR=/tmp/football-ai-mpl \
  .venv/bin/python tools/collect_ball_detector_predictions.py \
  --annotations data/ball_ground_truth/brandevoortBRAB_vid2/annotations.json \
  --threshold 0.05

.venv/bin/python tools/evaluate_ball_detection.py \
  --annotations data/ball_ground_truth/brandevoortBRAB_vid2/annotations.json \
  --predictions data/ball_ground_truth/brandevoortBRAB_vid2/detector_predictions.json
```

This exposes whether recall was already absent in RF-DETR or was lost in the
person-overlap filter. The same evaluator can separately compare the reviewed
frames with a ball-tracking report:

```bash
.venv/bin/python tools/evaluate_ball_detection.py \
  --annotations data/ball_ground_truth/brandevoortBRAB_vid2/annotations.json \
  --predictions output/ball/brandevoortBRAB_vid2_ball_tracking.json
```

For tracker JSON, only observations with `source="detected"` count as detector
predictions.
Predicted, interpolated, and stationary-held frames are deliberately excluded.
The JSON report contains precision, recall, F1, frame-level results, and
separate counts for visible misses, occluded misses, localization errors,
duplicates, and false positives on frames without a visible ball. Because the
ball is tiny, a prediction matches by either IoU or center distance; both
tolerances are explicit command-line parameters.

### First reviewed detector baseline

The human review of 21 frames from `brandevoortBRAB_vid2` frames 400-500 was
completed on 2026-08-02. Five frames show the active ball clearly, five show it
partly occluded by a player, and eleven contain no visible active ball.

At the 0.05 detector threshold, raw RF-DETR candidates match 9/10 visible or
occluded balls: 90.0% recall. Candidate-level precision is only 1.55% because
the same reviewed frames contain 573 false-positive candidates. This confirms
that lowering the threshold recovers real ball signal but cannot be used as a
standalone selection rule.

The original whole-person overlap filter retained 158 candidates but removed
all nine matching ball candidates, reducing recall to 0%. The filter is now
body-region-aware: compact weak candidates in the leg zone can reach the
temporal tracker, while head, torso, tiny shoe-texture, and oversized clothing
hits remain excluded. On the same reviewed frames it retains 336 candidates
and matches 8/10 visible or occluded balls: 80.0% recall and 2.38% precision.
The remaining errors are localization misses on frames 485 and 490. Frame 485
has no raw detector candidate within the agreed 20-pixel center tolerance;
frame 490 has a nearby raw candidate higher in the player box, but preserving
it would weaken the torso/shoe guard without enough evidence that it is the
ball. This small problem-window baseline therefore does not justify opening
the body filter further.

The physical tracker detections also score 0/10 on this five-frame sampling.
Predicted observations are deliberately excluded, so this result measures
direct anchors rather than trajectory coverage. These figures apply only to
the reviewed problem window and are not yet whole-clip performance claims.

### Player and team possession magnet

Ball position and possession confidence are intentionally separate. When the
physical ball disappears briefly, the possession tracker attaches an inferred
possession hypothesis to the last confirmed owner. The short occlusion window
is extended to at most 90 frames when that player is still tracked and the
local player cluster contains only recognized teammates. This covers cases
where several Brandevoort players surround an occluded ball without awarding
possession to an arbitrary visible object.

The team magnet remains `inferred`: it does not create a physical ball point,
a pass, or a turnover. A nearby same-team track may replace a disappeared
technical owner track without creating a pass, provided no opponent is inside
the same small handoff radius. A visible owner can remain inferred during a
crowded duel; proximity alone is not proof of a turnover. The magnet stops at
the bounded unseen limit, and a later detected ball remains authoritative.
The match-timeline layer preserves observations tagged with
`evidence="team_magnet"` as estimated possession, but does not use them to
confirm a team switch or derive an event.

On the reviewed `brandevoortBRAB_vid2` problem window, the last controlled
owner uses track 19. When that technical track disappears, a same-team proxy
handoff selects nearby track 48 without creating a pass. The canonical report
now retains 80 `team_magnet` frames between frames 400 and 500; samples from
frame 430 through frame 500 remain visibly labelled as estimated possession.
The window contains no derived pass or turnover. Frame 420 remains contested,
and later visible ball evidence remains authoritative. The possession QA is
rendered from the original video so an outdated or damaged derived entities
video cannot truncate visual validation.

## Known ambiguity: multiple balls

`TrainingClipXBOTGO_High` is a training scene with multiple balls, cones, white
shoes, and field markings visible at the same time. Geometric continuity can
select a plausible moving ball, but it cannot always establish which ball is
the active ball for the drill.

The first active-ball signal is now a bounded player-proximity tie-breaker for
initial acquisition. It can choose a similarly credible candidate near the
current play, while a clearly stronger ball in free flight still wins. Future
work can add possession continuity, a kick or player-contact event, and
coordinated movement of the surrounding players. A lower detector threshold on
its own is not sufficient: a 0.03 experiment increased early 15-second
visibility from 30.3% to 37.2%, mainly by interpolating between extra weak
anchors, without resolving active-ball identity.

A confirmed active-play handoff can replace a weak wrong track only when a
distant candidate scores at least 0.75, remains coherent for three consecutive
frames, and is supported by at least two nearby players. A credible candidate
near the existing trajectory blocks this handoff. On the first 15 seconds of
`brandevoortbrab`, the resulting QA run produced 308 detected, 16 predicted,
69 interpolated, and 55 stationary-held frames (99.8% visible). Visual checks
around frames 324-333 and 383-403 confirmed that the marker follows the
foreground ball during the duels instead of a weak candidate higher in the
player box or at the far side of the pitch.

Strong candidates at a player's feet receive priority when the existing
continuation is only weak. A sufficiently large candidate may therefore remain
eligible when it overlaps the bottom of a person box; this fixes the common
case where the real ball is partly hidden by a boot or leg. Small shoe-sized
candidates remain rejected, and a credible ball trajectory in flight blocks
the foot preference.

The person-overlap filter also preserves a large, strong ball that is only
partly visible beside a leg, while candidates inside the head, torso, lower
body, or shoe area are rejected. A confirmed player contact remains active for 15 frames, so a
real cut or turn can reverse the ball direction after a short occlusion. This
removed the former gap from frames 275-323 in `brandevoortbrab`; direct ball
detections now resume at frames 275 and 284 and continue through the turn.

On the first 30 seconds of `brandevoortbrab`, visual review found that 9.9% and
6.3% remote restarts at frames 699 and 895 were actually players' heads. A
remote restart now needs at least 30% confidence and ground-level footpoint
support; local continuation of an already proven track can still use the 5%
CLI threshold. After adding confirmed strong one-player restarts, the refreshed
run contains 381 detected, 21 predicted, 68 interpolated, and 57
stationary-held observations: 527/899 frames (58.6%). A new 52.0% restart at
frame 466 was visually confirmed on the active ball at a player's foot. The
false head segments at 699 and 895 remain absent.

The first 30 seconds of the separate `brandevoortBRAB_vid2` amateur clip were
regenerated with the same tracker on 2026-08-01. The run contains 105 detected,
103 predicted, 73 interpolated, and 200 stationary-held observations: 481/899
frames (53.5%). All observations remain in one track segment. Targeted visual
checks at the initial acquisition, the largest image-space jumps around frames
275 and 558, and the gap boundaries around frames 350, 430, 488, 602, 690, and
703 did not reveal a switch to a head, shoe, cone, or shadow. The result still
has material gaps: frames 430-487, 602-641, and 350-385 are uncovered, and the
track ends after frame 703 for the analyzed interval. This is therefore a
regression baseline, not evidence that detector recall or active-ball tracking
is complete.

An early-motion bootstrap additionally allows three coherent weak detections
after a strong player-supported anchor to establish velocity. During its short
lock-on window, physically plausible weak continuations may update that
velocity. On the first 12 seconds of `TrainingClipXBOTGO_High`, this increased
visible frames from 46/359 (12.8%) to 59/359 (16.4%). The remaining long gap is
primarily a detector-signal problem; an OpenCV MIL experiment was rejected
because it drifted within the nearby player cluster.

The QA path now also uses adaptive local RF-DETR inference after the first
credible anchor. On this 4K source, the same ball that scores around 0.05 in a
full-frame pass often scores 0.63-0.81 in the detail crop. Crop candidates
smaller than 12 pixels are rejected as texture. Stationary filling requires
stable three-detection clusters on both sides of a gap, and linear
interpolation is limited to 0.5 seconds so an airborne arc is not flattened
into a false stationary path. Together these changes raised the 30-second QA
visibility from 43.0% to 66.4%. This remains a tracker QA metric rather than a
claim of 66.4% detector recall or active-ball accuracy.
