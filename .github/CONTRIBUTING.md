# Contributing

Bug reports and pull requests are welcome. Keep changes small and focused;
match the existing terse register of the codebase.

## Running the tests

```bash
docker compose run --rm --entrypoint python3 analysis-engine tests/test_<name>.py
```

(e.g. `tests/test_classifier.py`.) The suite runs without any camera footage
present — footage-dependent tests (such as `tests/test_ground_truth.py`)
skip when their clips are not on disk. Verify your change passes the tests
that apply to it.

## Never commit footage or model weights

Camera recordings (`clips/`, `output/`, `recordings/` and common video
extensions) and model weights (`*.pt`, `*.onnx`) are gitignored. They are
large, they may contain private imagery, and they MUST NEVER be committed.

## Detection and tracking changes

If you change detection or tracking — model, confidence thresholds, tracker
profiles, the stationary-vehicle suppression rules — re-run the
detector/tracker sweep in `benchmarks/` and include the results in your pull
request. See `benchmarks/README.md` for the capture and sweep commands.
