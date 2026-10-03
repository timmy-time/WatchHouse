"""Ground-truth validation test suite for video event analysis engine."""

import os
import sys

from engine.pipeline import AnalysisPipeline

GROUND_TRUTH_CASES = [
    {
        "filename": "3_6_DVR0000_Camera C_30515ca9438a43dab33b4274723d7069_20260103170429.mp4",
        "expected_verdict": "KEEP",
        "description": "Person moving at camera c (putting on shoes)",
    },
    {
        "filename": "2_6_DVR0000_Camera B_a3473e0a29a64fe88d616d10a8614ef2_20260103145328.mp4",
        "expected_verdict": "KEEP",
        "description": "Person walking down sidewalk in blue jacket",
    },
    {
        "filename": "1_6_DVR0000_Camera A_136488f5ec8d4f0f8beef0396ac85ad3_20260103141555.mp4",
        "expected_verdict": "DISCARD",
        "description": "Daytime camera a with stationary parked vehicles",
    },
    {
        "filename": "1_6_DVR0000_Camera A_14dd2e8f7dff4fd5912ee261796386f9_20260103195427.mp4",
        "expected_verdict": "DISCARD",
        "description": "Night IR spiderweb in front of stationary parked car",
    },
    {
        "filename": "1_6_DVR0000_Camera A_0053ae04568a4cd8bc7daa1af203b1d5_20260103222915.mp4",
        "expected_verdict": "DISCARD",
        "description": "Night headlight reflection sweep across fence",
    },
]


def run_tests():
    clips_dir = "/data/clips/20260103"
    pipeline = AnalysisPipeline(gpus=[0], vid_stride=15, conf_threshold=0.30)

    pairs = []
    for case in GROUND_TRUTH_CASES:
        clip_path = os.path.join(clips_dir, case["filename"])
        thumb_path = clip_path.replace(".mp4", ".png")
        if not os.path.exists(thumb_path):
            thumb_path = None
        pairs.append((clip_path, thumb_path))

    print(f"Running ground truth validation on {len(pairs)} cases...")
    results = pipeline.process_batch(pairs)

    all_passed = True
    print("\n=== Validation Results ===")
    for case, res in zip(GROUND_TRUTH_CASES, results):
        verdict = res["verdict"]
        expected = case["expected_verdict"]
        passed = (verdict == expected)
        if not passed:
            all_passed = False
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {case['description']}")
        print(f"       File:     {case['filename']}")
        print(f"       Expected: {expected} | Actual: {verdict} ({res['primary_reason']})")
        print(f"       Tracks:   {len(res['detected_tracks'])} tracked objects\n")

    if not all_passed:
        print("ERROR: One or more ground truth tests failed!", file=sys.stderr)
        sys.exit(1)

    print("All ground truth validation tests passed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    run_tests()
