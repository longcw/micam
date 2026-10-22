"""Tests for the watchdog that recovers streams which stop carrying video."""
import unittest

from micam.watchdog import StreamWatchdog, video_packets


def snapshot(**packets):
    """A go2rtc /api/streams body with one video receiver per stream, or no producer for None."""
    streams = {}
    for name, count in packets.items():
        if count is None:
            streams[name] = {"producers": None}
        else:
            streams[name] = {"producers": [{"receivers": [
                {"id": 1, "codec": {"codec_type": "video"}, "packets": count},
                {"id": 2, "codec": {"codec_type": "audio"}, "packets": 999999},
            ]}]}
    return streams


class VideoPacketsTest(unittest.TestCase):
    def test_counts_video_only(self):
        self.assertEqual(video_packets(snapshot(s=5)["s"]), {1: 5})

    def test_no_producer(self):
        self.assertEqual(video_packets({"producers": None}), {})
        self.assertEqual(video_packets({}), {})


class StreamWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.dog = StreamWatchdog(["s1", "s2"], refresh_after=180, restart_after=360, restart_cooldown=1800)
        self.assertEqual(self.dog.observe(snapshot(s1=10, s2=10), 0), [])

    def test_healthy_streams_take_no_action(self):
        for t in range(15, 2000, 15):
            self.assertEqual(self.dog.observe(snapshot(s1=10 + t, s2=10 + t), t), [])

    def test_audio_alone_does_not_count_as_alive(self):
        streams = snapshot(s1=10, s2=10)
        streams["s1"]["producers"][0]["receivers"][1]["packets"] += 500
        self.assertEqual(self.dog.observe(streams, 200), ["refresh"])

    def test_escalates_refresh_then_restart_then_gives_up(self):
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=20), 179), [])
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=30), 180), ["refresh"])
        # the refresh is spent once per stall
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=40), 300), [])
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=50), 360), ["restart"])
        # after the restart a fresh refresh is allowed, but never a second restart
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=5), 540), ["refresh"])
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=6), 720), [])
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=7), 5000), [])

    def test_recovery_resets_the_escalation(self):
        self.dog.observe(snapshot(s1=None, s2=20), 180)
        self.dog.observe(snapshot(s1=None, s2=30), 360)
        self.assertEqual(self.dog.observe(snapshot(s1=3, s2=40), 400), [])
        # a later, separate stall starts over from a refresh
        self.assertEqual(self.dog.observe(snapshot(s1=3, s2=50), 580), ["refresh"])

    def test_new_producer_counts_as_video(self):
        streams = snapshot(s1=None, s2=20)
        self.dog.observe(streams, 100)
        moved = snapshot(s1=1, s2=30)
        moved["s1"]["producers"][0]["receivers"][0]["id"] = 77
        self.dog.observe(moved, 170)
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=40), 340), [])

    def test_restart_cooldown_spans_streams(self):
        self.dog.observe(snapshot(s1=None, s2=20), 180)
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=30), 360), ["restart"])
        self.dog.observe(snapshot(s1=5, s2=None), 400)
        self.dog.observe(snapshot(s1=6, s2=None), 580)
        # s2 stalls well inside the cooldown, so it waits rather than restarting again
        self.assertEqual(self.dog.observe(snapshot(s1=7, s2=None), 760), [])
        self.assertEqual(self.dog.observe(snapshot(s1=8, s2=None), 2160), ["restart"])

    def test_no_restart_while_miloco_is_starting(self):
        self.dog.observe(snapshot(s1=None, s2=20), 180)
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=30), 360, restart_allowed=False), [])
        self.assertEqual(self.dog.observe(snapshot(s1=None, s2=40), 375, restart_allowed=True), ["restart"])


if __name__ == "__main__":
    unittest.main()
