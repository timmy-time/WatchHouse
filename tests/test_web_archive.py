import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime

from fastapi.testclient import TestClient

from engine.web.app import create_app
from engine.web.archive import (
    build_facets,
    build_index,
    filter_indices,
    parse_class_filter,
    parse_date_bound,
    parse_dvr_timestamp,
    parse_tracks,
    sort_indices,
    track_class_counts,
)


def _track(class_name, track_id=1):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "frame_count": 10,
        "total_frames": 30,
        "avg_confidence": 0.7,
        "is_stationary": False,
    }


def _record(
    camera,
    timestamp,
    *,
    verdict="DISCARD",
    reason="stationary_vehicles_only",
    classes=("car",),
    confidence="0.70",
    duration="30",
    filename=None,
    tracks_as_list=False,
):
    name = filename or f"1_6_DVR0000_{camera}_{'0' * 32}_{timestamp or 'unknown'}.mp4"
    tracks = [_track(c, i + 1) for i, c in enumerate(classes)]
    return {
        "clip_path": f"/data/clips/{camera}/{name}",
        "thumbnail_path": f"/data/clips/{camera}/{name}.png",
        "filename": name,
        "camera": camera,
        "timestamp": timestamp or "",
        "verdict": verdict,
        "primary_reason": reason,
        "confidence": confidence,
        "duration_frames": duration,
        "stationary_vehicle_count": "1",
        "detected_tracks": tracks if tracks_as_list else repr(tracks),
    }


class ArchiveHelperTests(unittest.TestCase):
    def test_parse_dvr_timestamp_variants(self):
        self.assertEqual(parse_dvr_timestamp("20260103141555"), datetime(2026, 1, 3, 14, 15, 55))
        self.assertIsNone(parse_dvr_timestamp(""))
        self.assertIsNone(parse_dvr_timestamp(None))
        self.assertIsNone(parse_dvr_timestamp("not-a-timestamp"))
        self.assertIsNone(parse_dvr_timestamp("2026010"))
        self.assertIsNone(parse_dvr_timestamp("20261303141555"))  # month 13
        self.assertEqual(parse_dvr_timestamp(20260103141555), datetime(2026, 1, 3, 14, 15, 55))

    def test_parse_tracks_accepts_repr_and_list(self):
        expected = [_track("car")]
        self.assertEqual(parse_tracks(repr(expected)), expected)
        self.assertEqual(parse_tracks(expected), expected)
        self.assertEqual(parse_tracks("[{'class_name': 'dog'}, 'junk', 3]"), [{"class_name": "dog"}])
        self.assertEqual(parse_tracks("[not valid python"), [])
        self.assertEqual(parse_tracks(""), [])
        self.assertEqual(parse_tracks(None), [])
        self.assertEqual(parse_tracks({"class_name": "car"}), [])

    def test_track_class_counts_groups_case_insensitively(self):
        raw = repr([_track("Car", 1), _track("car", 2), _track("person", 3), _track("", 4)])
        self.assertEqual(track_class_counts(raw), {"car": 2, "person": 1})

    def test_parse_date_bound_day_semantics(self):
        self.assertEqual(parse_date_bound("2026-01-03", is_end=False), datetime(2026, 1, 3, 0, 0, 0))
        self.assertEqual(parse_date_bound("2026-01-03", is_end=True), datetime(2026, 1, 3, 23, 59, 59))
        self.assertEqual(parse_date_bound("2026-01-03T14:30", is_end=True), datetime(2026, 1, 3, 14, 30))
        self.assertEqual(parse_date_bound("2026-01-03 14:30:15", is_end=False), datetime(2026, 1, 3, 14, 30, 15))
        self.assertIsNone(parse_date_bound("", is_end=True))
        self.assertIsNone(parse_date_bound("yesterday", is_end=True))

    def test_parse_class_filter(self):
        self.assertEqual(parse_class_filter("car, Person ,dog"), ["car", "person", "dog"])
        self.assertEqual(parse_class_filter("car,,  "), ["car"])
        self.assertEqual(parse_class_filter(""), [])
        self.assertEqual(parse_class_filter(None), [])

    def _fixture(self):
        results = [
            _record("Camera A", "20260103141555", classes=("car", "truck"), confidence="0.70"),
            _record("Camera A", "20260103143000", classes=("person",), verdict="KEEP",
                    reason="high_value_object", confidence="0.95"),
            _record("Camera B", "20260104090000", classes=("car",), confidence="0.50"),
            _record("Camera B", "20260104120000", classes=("dog",), verdict="KEEP",
                    reason="high_value_object", confidence="0.80", duration="90"),
            _record("Camera C", "20260102120000", classes=("car",), confidence="0.60"),  # oldest camera
        ]
        return results, build_index(results)

    def test_filter_combinations(self):
        results, index = self._fixture()

        self.assertEqual(filter_indices(results, index), [0, 1, 2, 3, 4])
        self.assertEqual(filter_indices(results, index, camera="Camera B"), [2, 3])
        self.assertEqual(filter_indices(results, index, verdict="KEEP"), [1, 3])
        self.assertEqual(filter_indices(results, index, classes=["PERSON"]), [1])
        self.assertEqual(filter_indices(results, index, classes=["car", "dog"]), [0, 2, 3, 4])
        self.assertEqual(filter_indices(results, index, reason="stationary"), [0, 2, 4])
        self.assertEqual(
            filter_indices(results, index, camera="Camera B", classes=["dog"], verdict="KEEP"), [3]
        )

    def test_filter_date_range_includes_whole_end_day(self):
        results, index = self._fixture()
        day = parse_date_bound("2026-01-03", is_end=False)
        day_end = parse_date_bound("2026-01-03", is_end=True)
        self.assertEqual(filter_indices(results, index, date_from=day, date_to=day_end), [0, 1])

        # Records whose clock is unknown cannot satisfy a date bound.
        self.assertEqual(
            filter_indices(results, index, date_from=parse_date_bound("2020-01-01", False)), [0, 1, 2, 3, 4]
        )

    def test_sort_keys(self):
        results, index = self._fixture()
        all_idx = [0, 1, 2, 3, 4]

        # Default: newest first, ties by original position.
        self.assertEqual(sort_indices(results, index, all_idx), [3, 2, 1, 0, 4])
        self.assertEqual(sort_indices(results, index, all_idx, "date_asc"), [0, 1, 2, 3, 4])
        self.assertEqual(sort_indices(results, index, all_idx, "bogus"), [3, 2, 1, 0, 4])

        # Camera sort: name asc/desc; newest-first tiebreak inside each camera.
        self.assertEqual(sort_indices(results, index, all_idx, "camera"), [1, 0, 4, 3, 2])
        self.assertEqual(sort_indices(results, index, all_idx, "camera_desc"), [3, 2, 4, 1, 0])

        # Class sort: alphabetical by the record's first class, records without classes last.
        results_no_class, index_no_class = self._fixture()
        results_no_class.append(_record("Camera A", "20260105000000", classes=()))
        index_no_class = build_index(results_no_class)
        ordered = sort_indices(results_no_class, index_no_class, list(range(len(results_no_class))), "class")
        self.assertEqual(ordered, [2, 0, 4, 3, 1, 5])

        self.assertEqual(sort_indices(results, index, all_idx, "confidence_desc"), [1, 3, 0, 4, 2])
        self.assertEqual(sort_indices(results, index, all_idx, "duration_desc"), [3, 2, 1, 0, 4])

    def test_sort_is_stable_for_equal_keys(self):
        results = [
            _record("Camera A", "20260103000000", confidence="0.5"),
            _record("Camera A", "20260103000000", confidence="0.5"),
            _record("Camera A", "20260103000000", confidence="0.5"),
        ]
        index = build_index(results)
        self.assertEqual(sort_indices(results, index, [0, 1, 2], "camera"), [0, 1, 2])
        self.assertEqual(sort_indices(results, index, [0, 1, 2], "confidence_desc"), [0, 1, 2])

    def test_build_facets(self):
        results, index = self._fixture()
        facets = build_facets(results, index)
        self.assertEqual(facets["cameras"], {"Camera A": 2, "Camera B": 2, "Camera C": 1})
        self.assertEqual(facets["classes"], {"car": 3, "dog": 1, "person": 1, "truck": 1})
        self.assertEqual(facets["date_min"], "2026-01-03T14:15:55")
        self.assertEqual(facets["date_max"], "2026-01-04T12:00:00")

    def test_build_facets_without_dates(self):
        results = [_record("Camera A", "", classes=())]
        facets = build_facets(results, build_index(results))
        self.assertIsNone(facets["date_min"])
        self.assertIsNone(facets["date_max"])
        self.assertEqual(facets["classes"], {})


class ArchiveApiTests(unittest.TestCase):
    def setUp(self):
        # create_app installs BasicAuthMiddleware when both vars are set, which would
        # turn every request here into a 401 in an environment that exports them.
        self._auth_env = {
            key: os.environ.pop(key, None) for key in ("DASHBOARD_USER", "DASHBOARD_PASSWORD")
        }
        self.tmp = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmp, "output")
        self.clips_dir = os.path.join(self.tmp, "clips")
        os.makedirs(os.path.join(self.output_dir, "live"), exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)

        results = [
            _record("Camera A", "20260103141555", classes=("car", "truck"), confidence="0.70"),
            _record("Camera A", "20260103143000", classes=("person",), verdict="KEEP",
                    reason="high_value_object", confidence="0.95"),
            _record("Camera B", "20260104090000", classes=("car",), confidence="0.50"),
            _record("Camera B", "20260104120000", classes=("dog",), verdict="KEEP",
                    reason="high_value_object", confidence="0.80"),
            _record("Camera C", "20260102120000", classes=("car",), confidence="0.60"),
        ]
        with open(os.path.join(self.output_dir, "analysis_results.json"), "w") as fh:
            json.dump(
                {
                    "total_clips": len(results),
                    "kept_count": 2,
                    "discarded_count": 3,
                    "error_count": 0,
                    "reduction_percentage": 40.0,
                    "runtime_seconds": 1.0,
                    "results": results,
                },
                fh,
            )
        self.app = create_app(self.output_dir, self.clips_dir, os.path.join(self.tmp, "live.yaml"))
        self.client = TestClient(self.app)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for key, value in self._auth_env.items():
            if value is not None:
                os.environ[key] = value

    def test_summary_exposes_facets(self):
        body = self.client.get("/api/archive/summary").json()
        self.assertEqual(body["total_clips"], 5)
        self.assertEqual(body["cameras"], {"Camera A": 2, "Camera B": 2, "Camera C": 1})
        self.assertEqual(body["classes"]["car"], 3)
        self.assertEqual(body["date_min"], "2026-01-03T14:15:55")
        self.assertEqual(body["date_max"], "2026-01-04T12:00:00")

    def test_list_is_newest_first_with_derived_fields(self):
        body = self.client.get("/api/archive").json()
        self.assertEqual(body["total"], 5)
        self.assertEqual([i["idx"] for i in body["items"]], [3, 2, 1, 0, 4])
        first = body["items"][0]
        self.assertEqual(first["datetime"], "2026-01-04T12:00:00")
        self.assertEqual(first["date"], "2026-01-04")
        self.assertEqual(first["time"], "12:00:00")
        self.assertEqual(first["classes"], ["dog"])
        self.assertEqual(first["class_counts"], {"dog": 1})
        self.assertNotIn("detected_tracks", first)

    def test_filters_and_total_reflect_filtering(self):
        body = self.client.get("/api/archive", params={"camera": "Camera B", "limit": 1}).json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(len(body["items"]), 1)
        self.assertEqual(body["items"][0]["camera"], "Camera B")

        body = self.client.get("/api/archive", params={"class_name": "person"}).json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["items"][0]["classes"], ["person"])

        body = self.client.get("/api/archive", params={"verdict": "KEEP", "class_name": "dog"}).json()
        self.assertEqual(body["total"], 1)

        body = self.client.get("/api/archive", params={"date_from": "2026-01-04", "date_to": "2026-01-04"}).json()
        self.assertEqual(body["total"], 2)
        self.assertEqual({i["camera"] for i in body["items"]}, {"Camera B"})

        body = self.client.get("/api/archive", params={"reason": "high_value"}).json()
        self.assertEqual(body["total"], 2)

    def test_sort_param(self):
        body = self.client.get("/api/archive", params={"sort": "camera"}).json()
        self.assertEqual([i["camera"] for i in body["items"]], ["Camera A", "Camera A", "Camera B", "Camera B", "Camera C"])

        body = self.client.get("/api/archive", params={"sort": "confidence_desc"}).json()
        self.assertEqual([i["confidence"] for i in body["items"]], ["0.95", "0.80", "0.70", "0.60", "0.50"])

    def test_item_endpoint_includes_derived_fields(self):
        item = self.client.get("/api/archive/1").json()
        self.assertEqual(item["datetime"], "2026-01-03T14:30:00")
        self.assertEqual(item["classes"], ["person"])

    def test_unknown_clip_index_returns_404(self):
        self.assertEqual(self.client.get("/api/archive/99").status_code, 404)


if __name__ == "__main__":
    unittest.main()
