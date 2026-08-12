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

To tighten existing detector-visible boxes without stepping through occluded or
not-visible frames, use the dedicated recheck queue:

```bash
.venv/bin/python tools/review_ball_ground_truth.py \
  --annotations data/ball_ground_truth/ClipTrainingBallFar_dense_900_960/annotations.json \
  --recheck-visible
```

This mode contains only existing `human_reviewed` + `visible` annotations with
a box. Saving advances to the next item in that fixed queue; it does not change
labels outside the queue.

Click the center of the active ball to place a 20-pixel box; `+` and `-` resize
the current box in two-pixel steps. Mouse dragging remains available when a
custom box is easier. The magnified preview stays centered on the current box
and draws that same box over the enlarged pixels, including its exact width and
height in the sidebar. Use the
on-screen buttons to save the ball as visible, player-occluded, or not visible;
the next frame opens automatically. Keyboard shortcuts remain available, but
are not required. The cursor magnifier helps with small distant balls. Every
confirmation is saved immediately; `A` and `D` navigate between frames. Only
entries explicitly saved by this interface as `human_reviewed` participate in
the metrics.

Annotate the complete visible ball appearance, not an idealized circle. A ball
can deform at impact and motion blur or frame exposure can make it elongated.
In those frames, drag a tight rectangular box around the full visible/deformed
or blurred footprint; do not force it back to a square. Detector proposals and
later candidate filters must likewise treat aspect ratio as soft evidence, not
as a hard roundness requirement.
When a partial manifest is reopened, the reviewer resumes at the first
unreviewed frame. After confirmation it skips already reviewed frames and
continues with the next open item; `A` and `D` still provide sequential
navigation for deliberate rechecks.
Candidate prefill is deliberately limited to detections with at least 0.75
confidence. On the reviewed `brandevoortBRAB_vid2` frames, simply choosing the
highest-scoring low-confidence candidate selected the active ball on only 3
of 26 unique reviewed ball frames. The reviewed `brandevoortbrab_clip3` proposals
all exceeded the cutoff and matched the active ball in 13/13 dense frames.
Weak candidates therefore remain visible to the analysis pipeline but are not
presented as trustworthy green review shortcuts.

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
all nine matching ball candidates, reducing recall to 0%. A later
body-region-aware version still made an irreversible decision before temporal
or active-ball selection. Dense review of a goalkeeper holding the ball showed
the same failure mode outside the leg zone: person filtering reduced raw
detector recall from 23.7% to 10.2%. Detector candidates now remain available
even when they overlap a person box. This applies to goalkeepers and field
players because control, tackles, headers, and throw-ins can all put the real
ball inside a person box. Body overlap is context for later selection, not
proof that a candidate is false.

This change deliberately increases clutter. On the reviewed goalkeeper window
raw candidate precision is only 1.4%, so preserving evidence is not the same as
selecting the active ball. Fresh candidates are required after this policy
change; older person-filtered caches are invalid for comparison.

The physical tracker detections also score 0/10 on this five-frame sampling.
Predicted observations are deliberately excluded, so this result measures
direct anchors rather than trajectory coverage. These figures apply only to
the reviewed problem window and are not yet whole-clip performance claims.

### Goalkeeper possession baseline

`ClipTrainingBallFar` frames 900-960 contain 61 human-reviewed goalkeeper
possession frames: 45 visible, 14 occluded, and two not visible. With person
overlap filtering removed, raw RF-DETR candidates match 14/59 visible or
occluded annotations. All 14 matches occur on frames 947-960; 32 reviewed ball
frames have no matching detector candidate. The current tracker follows a
different visual object through this window and scores precision, recall, and
F1 of 0.0 against the dense annotations.

Adding this clip as a fourth source does not rescue the patch classifier. Its
leave-one-video-out F1 is 0.0 on both the goalkeeper clip and
`brandevoortBRAB_vid2`; mean F1 falls to 0.184. Keep the classifier out of
`BallTracker`. The next improvement must address far/held-ball detector recall
and then prove active-ball selection on clip-separated human review. A
goalkeeper possession magnet may provide bounded inferred possession, but may
not invent a physical ball position when the detector has no evidence.

A CPU follow-up added label-free continuity features: confidence rank,
nearby-frame displacement, size/confidence change, and midpoint consistency.
Run it with `train_active_ball_classifier.py --feature-set patch-temporal`.
Mean leave-one-video-out F1 increased only from 0.184 to 0.210. Both difficult
holdouts remained at F1 0.0, while `ClipTrainingBallFar` produced 1,154 false
positives. This is a rejected experiment, not a tracking improvement; do not
integrate the temporal classifier into `BallTracker`.

### Detector fine-tuning data

`tools/export_ball_detector_dataset.py` exports human-reviewed frames in the
Roboflow COCO directory format consumed by the installed RF-DETR version. It
requires one or more explicit `--validation-source` values so neighbouring
frames from a validation video cannot leak into training. Only human-reviewed
`visible` boxes become positive detector annotations. Human-reviewed
`not_visible` frames are exported as negative images. `occluded` and
`unreviewed` frames are excluded because an inferred occluded box is not proof
of detector-visible pixels.

The first keeper-holdout export is
`data/ball_detector_rfdetr_keeper_holdout`: 113 training images from four
source videos (89 positive, 24 negative) and 47 validation images from
`ClipTrainingBallFar` (45 positive, two negative). RF-DETR 1.8.3 recognizes the
directory as a valid dataset. This is enough for a controlled experiment, not
enough to claim a robust detector: the training set is small and the keeper
review used fixed 20x20 boxes. The optional RF-DETR training dependencies are
installed. Use the guarded training entry point below on a CUDA or MPS machine;
it refuses an accidental full CPU run unless `--allow-cpu` is explicit:

```bash
.venv/bin/python tools/train_ball_detector.py \
  --dataset data/ball_detector_rfdetr_keeper_holdout \
  --output output/ball/rfdetr_keeper_holdout
```

RF-DETR reinitializes its detection head when changing from the pretrained 90
classes to the single `sports ball` class. For the small keeper-holdout export,
the default early-stopping window can therefore end before the new head has had
enough epochs to learn. Use `--disable-early-stopping` only for the controlled
full-length comparison; keep the clip-separated validation source unchanged.

The controlled RTX 3090 comparison completed all 50 epochs with early stopping
disabled. It did not validate the fine-tuning approach. The best regular result
was already epoch 1: mAP 50:95 0.00119, precision 0.00480, recall 0.28889 and F1
0.00944; best EMA mAP 50:95 was 0.00135. By epoch 50 all reported validation
metrics had collapsed to zero. Keep this checkpoint out of the detector and
tracker. The run artifacts are in
`output/ball/rfdetr_keeper_holdout_50epochs`; more epochs on this export are not
a justified next step.

The full-frame export also exposes a scale mismatch: at RF-DETR Medium's 576px
square input, the fixed keeper boxes become about 3x5.3 pixels, versus a mean
of roughly 11x19.6 pixels in training. A leakage-safe tiled export is available
through `export_ball_detector_dataset.py --tile-size 960 --tile-overlap 0.25`.
It preserves source-video separation and skips any tile that would cut through
a labelled ball. The first tile export contains 226 training tiles (167
positive) and 705 keeper-holdout tiles (90 positive); keeper boxes become
12x12 pixels at model input. Evaluation must still scan every full holdout
frame with the same tile grid and merge overlaps with NMS. A crop selected from
the known ground-truth location is not a valid holdout evaluation.

Use `tools/evaluate_ball_detector_checkpoint.py` to compare the pretrained and
fine-tuned models with identical full-frame COCO metrics. Its optional
`--tile-size 960 --tile-overlap 0.25` mode performs full-image tiled inference
and class-agnostic NMS before scoring against the original full-frame truth.

The exact 47-image keeper holdout comparison rejects the original full-frame
fine-tune: pretrained full-frame RF-DETR reaches mAP 50:95 0.00531 and mAR
0.0667, while its best fine-tuned checkpoint reaches only 0.00083 and 0.0822.
Tiling the pretrained detector raises mAR to 0.1467 but lowers mAP to 0.00117.

A ten-epoch RTX 3090 run on the tiled export also fails acceptance. Its best
regular checkpoint reaches mAP 50:95 0.00262 and mAR 0.2533 on the original 47
full frames, after tiled scanning and NMS, but emits 66,957 predictions. The
best EMA checkpoint reaches mAP 0.00164 and mAR 0.4222 while emitting 73,352
predictions. The one-epoch smoke checkpoint had mAP 0.00209, mAR 0.3978 and
35,050 predictions. Thus tiling exposes more true balls, but it amplifies
clutter instead of learning a usable ball detector. None of these checkpoints
belongs in `BallTracker`; more epochs on the same small, fixed-box dataset are
not justified.

The 45 keeper holdout boxes were subsequently rechecked by a human with the
magnified box overlay. All fixed 20x20 boxes were replaced by tight boxes from
12x12 through 18x18 pixels (26 are 14x14). Re-evaluation against this corrected
truth makes the rejection stronger. Pretrained full-frame RF-DETR reaches mAP
50:95 0.00519 and mAR 0.0578. Pretrained tiled inference reaches mAP 0.00168
and mAR 0.1844. The full-frame fine-tune falls to mAP 0.000011 and mAR 0.0044;
the one-epoch tiled model reaches 0.00056 and 0.1467; the ten-epoch regular and
EMA models reach respectively 0.00094/0.0711 and 0.00034/0.0933. The training
split did not change, so retraining those experiments would be identical and
is not warranted. Earlier fixed-box metrics above are retained only as a record
of the experiment and must not be used as the current acceptance baseline.

Two additional independent sources were then human-reviewed: 55 visible balls
from `ClipPSVAJAX` and 53 from `TrainingClipXBOTGO_High`. The resulting
six-source tiled training split has 437 positive and 941 negative tiles. A
one-epoch single-class smoke run improved over the earlier fine-tunes but still
failed the original-frame keeper holdout (mAP 50:95 0.00357, mAR 0.1578, and
140,090 predictions at threshold 0.001). At threshold 0.02 it retained 915
predictions with mAP 0.00334 and mAR 0.0911. A five-epoch follow-up regressed to
mAP 0.00113 and mAR 0.1111, so later epochs are rejected.

RF-DETR normally reinitialized its 90-class detection head for the one-class
export. A controlled `--preserve-coco-head` export kept 90 contiguous logit
slots and aligned `sports ball` with pretrained COCO slot 37; the reinitializing
warning disappeared during training. This did not rescue the approach. Its
one-epoch checkpoint scored mAP 0.000016 and mAR 0.0044 on the same original
keeper frames, and found nothing above threshold 0.02. Preserving the head alone
is therefore rejected too. A future detector experiment needs a different
optimization/model strategy, not more RF-DETR epochs on these layouts.

A YOLO26s tiny-object baseline reused the same six-source 960px tiles and the
same keeper holdout. COCO boxes were converted losslessly to YOLO labels,
including empty negative labels and non-square boxes. Training used 960px
input, the pretrained sports-ball classification row, AdamW at learning rate
0.001, one warmup epoch, mosaic 0.5, and 20% motion/Gaussian blur augmentation.
The ten-epoch validation peaked at epoch 3 and then regressed. On the 47
original keeper frames, tiled inference plus NMS reached only mAP 50:95
0.000118 and mAR 0.0067 at threshold 0.001 (1,164 predictions). Threshold 0.01
left 199 predictions with the same negligible metrics; thresholds from 0.05
up found no correct balls. This YOLO26s setup is rejected and must not be
integrated into `BallTracker`.

The failed architectures share a measurable data-domain problem. Run
`tools/analyze_ball_detector_domain.py` on a full-frame COCO export to report
box size, aspect ratio, sharpness, foreground contrast, local variation, and
brightness per split and source. On the six-source export, the 197 training
balls have median brightness 171.4 and foreground contrast 48.8, while the 45
keeper-holdout balls measure only 92.3 and 27.6. Median size differs less
severely (18x18 versus 14x14 pixels). Visual inspection confirms that the
holdout ball is darker, greener/greyer, and less distinct from the pitch.
Almost every current box is square (median aspect ratio 1.0 in both splits),
so the reviewed data also does not yet represent genuinely elongated
motion-blurred balls even though the review tool permits tight rectangular
boxes. This evidence points to missing real dark, low-contrast, and elongated
examples; brightness augmentation alone is not proof that the gap is closed.
Keep `ClipTrainingBallFar` untouched as the benchmark and acquire an
independent source before moving any comparable keeper footage into training.

A 60-frame sparse review spread over the full 14.6 minutes of
`brandevoortbrab.mov` added 34 visible balls, eight occluded frames, and 18
human-confirmed negatives. The visible balls are useful tiny-object examples
(median 8x8 pixels) and their median foreground contrast of 30.6 approaches
the keeper holdout's 27.6. They are not a dark-domain solution: their median
brightness is 148.2 versus 92.3 on the holdout, and all 34 reviewed boxes have
aspect ratio 1.0. The expanded full-frame dataset contains 231 training
positives and 45 negatives; its median training brightness moves only from
171.4 to 165.9. Keep this data, but do not interpret it as closing the dark or
elongated-ball gaps. The two long Brandevoort files also appear to show the
same match and camera, so treating them as independent domains would
overstate diversity.

Validate the split and resolved device without loading or training the model
with `--dry-run`.

### Dense review around the first track switch

A second human review covers every frame from 370 through 410 in the same
clip. It contains 41 reviewed frames: six with a visible active ball, eleven
with a player-occluded ball, and 24 where the active ball is not visible. This
dense window identifies the transition that a five-frame sample cannot show.

At the 0.05 threshold, both the raw detector and the person-filtered detector
match all 17 visible or occluded balls: 100% recall. Candidate-level precision
remains low (1.06% raw and 1.58% after the person filter), so these candidates
still require temporal selection.

The tracker creates one correct direct anchor at frame 370, then creates false
direct anchors on frames 379, 384, and 391 while the human annotations mark the
active ball as not visible. Its direct-detection precision is 25.0% and recall
is 5.9% in this dense window. By frame 387, when the real ball becomes visible
again, the selected trajectory is already about 59 pixels away.

The correct and incorrect candidates overlap in confidence, size, temporal
persistence, and player proximity. Correct candidates range from roughly
5.3-32.2% confidence, while the three false anchors score 7.8%, 31.6%, and
37.4%. A confidence, size, or player-distance threshold alone therefore cannot
fix this switch without removing genuine ball evidence. The next selection
improvement needs an additional identity signal, or must keep ambiguous weak
candidates as non-physical support until stronger evidence appears.

### Replaying tracker policy without detector inference

A normal `analyze_ball.py` run now stores the filtered per-frame candidates,
player context, and camera transforms in
`output/ball/<video>_ball_candidates.json`. The cache is a generated QA
artifact and is not versioned. Re-run only the tracker and QA render with:

```bash
MPLCONFIGDIR=/tmp/football-ai-mpl \
  .venv/bin/python tools/analyze_ball.py \
  --video brandevoortBRAB_vid2.mp4 \
  --threshold 0.05 \
  --reuse-candidates
```

The cache is accepted only for the same source-video path. A missing, empty,
unknown-version, or mismatched cache fails explicitly. This makes tracker
experiments repeatable and reduces a 30-second reference replay from minutes
of detector inference to roughly ten seconds on the checked development
machine. A normal run remains required after detector, person-filter, local
crop, or camera-motion changes.

Cache schema 2 additionally stores ByteTrack player IDs, current team labels,
footpoints, and boxes. Schema 1 caches remain readable, but naturally contain
no stable player identity. `BallTracker` records the nearest confirmed owner
track and team when a physical ball anchor is close to a tracked footpoint.
On the dense frames 370-410 this context alone does not improve direct-anchor
precision: the false candidate is coherent at the foot of the same likely
owner and is therefore indistinguishable from a partly occluded ball without
an additional visual identity feature. Player identity must not be presented
as proof that the candidate itself is the ball.

Candidate training labels can be derived from reviewed manifests with
`tools/build_active_ball_dataset.py`. Overlapping manifests must agree by
default. Open placeholders in a later dense manifest never replace an earlier
human-reviewed annotation; a later human review automatically replaces an
open placeholder. If a later, denser human review intentionally corrects an
earlier label, pass `--allow-conflicting-overrides`; the output records every
replaced frame
in `overridden_annotation_frames` so the correction remains auditable. The
current one-clip dataset report is deliberately blocked for training until it
contains at least 100 positive frames from at least three source clips.
Build each clip report separately, then combine the reports with
`tools/aggregate_active_ball_datasets.py`. The aggregator counts distinct
`source_video` values and rejects duplicate clips. Only sources with at least
one positive frame count toward the multi-clip readiness requirement, so an
empty or fully unreviewed manifest cannot make a dataset appear trainable.
Candidates matching the reviewed ball are positive. Non-matches at least 60
pixels from the reviewed ball are safe negatives; closer non-matches remain
ambiguous to avoid teaching the model that a duplicate or slightly misplaced
ball detection is background. Candidates on reviewed `not_visible` frames are
always negative.
Only reviewed `visible` frames are required to have a matching positive
candidate. An `occluded` annotation records the inferred ball location but
does not prove that detector-visible ball pixels exist; missing matches on
those frames are reported separately as `occluded_without_positive_frames`
and must not be fixed by loosening the geometric match threshold.

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
