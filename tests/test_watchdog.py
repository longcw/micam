"""Tests for the watchdog that ends a session publishing nothing."""
import unittest
from unittest import mock

from micam import PUBLISH_STALL_TIMEOUT, RTSPBridge


class Exited(Exception):
    """Stands in for os._exit, which a test cannot let through."""


def build_bridge(**kwargs):
    params = dict(
        base_url="https://miloco:8338",
        username="admin",
        password="secret",
        camera_id="123",
        rtsp_url="rtsp://host:8554/stream1",
    )
    params.update(kwargs)
    return RTSPBridge(**params)


class WatchPublishingTest(unittest.TestCase):
    def watch(self, bridge, now, rounds=3):
        """Run the loop until it exits, or until sleep has been called ``rounds`` times."""
        calls = []
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > rounds:
                raise Exited("loop did not exit")

        def exit_(code):
            calls.append(("exit", code))
            raise Exited(code)

        with mock.patch("micam.time.sleep", sleep), \
                mock.patch("micam.time.monotonic", return_value=now), \
                mock.patch("micam.faulthandler.dump_traceback",
                           lambda: calls.append(("dump", None))), \
                mock.patch("micam.threading.Timer") as timer, \
                mock.patch("micam.logger.error"), \
                mock.patch("micam.os._exit", exit_):
            with self.assertRaises(Exited):
                bridge.watch_publishing()
        return calls, timer, sleeps

    def test_exits_once_video_stops_reaching_ffmpeg(self):
        bridge = build_bridge()
        bridge._last_write = 0.0
        calls, _, _ = self.watch(bridge, now=PUBLISH_STALL_TIMEOUT + 1)
        self.assertIn(("exit", 1), calls)

    def test_a_recent_write_is_not_a_stall(self):
        bridge = build_bridge()
        bridge._last_write = 0.0
        calls, _, sleeps = self.watch(bridge, now=PUBLISH_STALL_TIMEOUT - 1)
        self.assertEqual(calls, [])
        self.assertEqual(len(sleeps), 4)

    def test_a_bridge_that_has_not_started_is_left_alone(self):
        # a slow login or a late first keyframe is covered by the socket deadlines,
        # so the watchdog must not count it as a stall and restart-loop the bridge
        bridge = build_bridge()
        calls, _, _ = self.watch(bridge, now=1e6)
        self.assertEqual(calls, [])

    def test_teardown_is_not_a_stall(self):
        bridge = build_bridge()
        bridge._last_write = 0.0
        bridge._shutting_down = True
        calls, _, _ = self.watch(bridge, now=PUBLISH_STALL_TIMEOUT + 1)
        self.assertEqual(calls, [])

    def test_the_exit_is_armed_before_anything_is_written(self):
        # a full stderr pipe is one of the stalls this catches, so a dump that blocks
        # must not be able to keep the process alive
        bridge = build_bridge()
        bridge._last_write = 0.0
        calls, timer, _ = self.watch(bridge, now=PUBLISH_STALL_TIMEOUT + 1)
        timer.assert_called_once()
        delay, _target, args = timer.call_args[0]
        self.assertGreater(delay, 0)
        self.assertEqual(args, (1,))
        timer.return_value.start.assert_called_once_with()
        self.assertEqual(calls[0], ("dump", None))

    def test_the_stack_dump_precedes_the_exit(self):
        bridge = build_bridge()
        bridge._last_write = 0.0
        calls, _, _ = self.watch(bridge, now=PUBLISH_STALL_TIMEOUT + 1)
        self.assertEqual(calls, [("dump", None), ("exit", 1)])


if __name__ == "__main__":
    unittest.main()
