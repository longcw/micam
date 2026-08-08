"""Tests for audio codec selection and the FFmpeg command it produces."""
import os
import unittest
from unittest import mock

from micam import (
    AUDIO_GIVEUP_SECONDS,
    AUDIO_INPUTS,
    AUDIO_RATES,
    AUDIO_PACE_INTERVAL,
    AUDIO_QUEUE_MAX_SECONDS,
    AUDIO_QUEUE_TARGET_SECONDS,
    AUDIO_BURST_GAP,
    AUDIO_GAP_MAX,
    AUDIO_GAP_MIN,
    AUDIO_RATE_MEASURE_TIMEOUT,
    AUDIO_RATE_MIN_PLAUSIBLE,
    AUDIO_RATE_SAMPLE_FRAMES,
    AUDIO_RECONNECT_DELAY,
    AUDIO_SILENCE,
    SHUTDOWN_TIMEOUT,
    STDERR_READ_TIMEOUT,
    RTSPBridge,
)


def audio_input(codec_id, rate=8000):
    """(format, rate, channels) as _open_audio assembles it after measuring."""
    fmt, channels = AUDIO_INPUTS[codec_id]
    return fmt, rate, channels


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


class StartFfmpegTest(unittest.TestCase):
    def start(self, bridge, audio_input):
        with mock.patch("subprocess.Popen") as popen, \
                mock.patch("os.pipe", return_value=(7, 8)), \
                mock.patch("os.close"):
            bridge._start_ffmpeg(audio_input)
            return popen.call_args[0][0], popen.call_args[1]

    def test_video_only_has_no_audio_mapping(self):
        cmd, kwargs = self.start(build_bridge(), None)
        self.assertIn("-i", cmd)
        self.assertIn("pipe:0", cmd)
        self.assertEqual(cmd.count("-i"), 1)
        self.assertNotIn("-c:a", cmd)
        self.assertNotIn("1:a", cmd)
        self.assertEqual(kwargs["pass_fds"], ())

    def test_g711a_adds_a_second_input_and_copies_it(self):
        bridge = build_bridge()
        cmd, kwargs = self.start(bridge, audio_input(1027))
        self.assertEqual(cmd.count("-i"), 2)
        self.assertIn("alaw", cmd)
        self.assertIn("pipe:7", cmd)
        self.assertEqual(cmd[cmd.index("-ar") + 1], "8000")
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")
        self.assertIn("1:a", cmd)
        self.assertEqual(kwargs["pass_fds"], (7,))
        self.assertEqual(bridge.audio_fd, 8)

    def test_g711u_selects_mulaw(self):
        cmd, _ = self.start(build_bridge(), audio_input(1026))
        self.assertIn("mulaw", cmd)

    def test_video_input_precedes_audio_input(self):
        cmd, _ = self.start(build_bridge(), audio_input(1027))
        # -map 0:v / -map 1:a rely on this order
        self.assertLess(cmd.index("pipe:0"), cmd.index("pipe:7"))

    def test_interleave_wait_is_disabled_when_audio_is_present(self):
        # a live stream gains nothing from the muxer holding video back for audio
        cmd, _ = self.start(build_bridge(), audio_input(1027))
        self.assertEqual(cmd[cmd.index("-max_interleave_delta") + 1], "0")

    def test_no_interleave_flag_without_audio(self):
        cmd, _ = self.start(build_bridge(), None)
        self.assertNotIn("-max_interleave_delta", cmd)

    def test_only_video_is_stamped_from_the_wallclock(self):
        cmd, _ = self.start(build_bridge(), audio_input(1027))
        # stamping raw PCM by arrival time compresses bursts and runs audio ahead of video
        self.assertEqual(cmd.count("-use_wallclock_as_timestamps"), 1)
        self.assertLess(cmd.index("-use_wallclock_as_timestamps"), cmd.index("pipe:0"))


class AudioInputTableTest(unittest.TestCase):
    def test_opus_and_pcm_are_not_muxed_yet(self):
        self.assertNotIn(1032, AUDIO_INPUTS)
        self.assertNotIn(1024, AUDIO_INPUTS)

    def test_g711_pair_is_eight_kilohertz_mono(self):
        for codec_id in (1026, 1027):
            _, channels = AUDIO_INPUTS[codec_id]
            self.assertEqual(channels, 1)


class CloseAudioFdTest(unittest.TestCase):
    def test_close_is_idempotent(self):
        bridge = build_bridge()
        bridge.audio_fd = 9
        with mock.patch("os.close") as close:
            bridge._close_audio_fd()
            bridge._close_audio_fd()
            close.assert_called_once_with(9)
        self.assertIsNone(bridge.audio_fd)

    def test_close_survives_an_already_closed_fd(self):
        bridge = build_bridge()
        bridge.audio_fd = 9
        with mock.patch("os.close", side_effect=OSError):
            bridge._close_audio_fd()
        self.assertIsNone(bridge.audio_fd)


class EnableAudioTest(unittest.TestCase):
    def test_defaults_to_enabled(self):
        self.assertTrue(build_bridge().enable_audio)

    def test_can_be_disabled(self):
        self.assertFalse(build_bridge(enable_audio=False).enable_audio)


class AudioFailureTest(unittest.TestCase):
    def test_audio_loss_terminates_ffmpeg(self):
        # ffmpeg cannot withdraw an announced audio track, so losing audio has to
        # restart the bridge rather than publish a track that never carries data
        bridge = build_bridge()
        proc = mock.Mock()
        proc.poll.return_value = None
        bridge.process = proc
        bridge._terminate_ffmpeg()
        proc.terminate.assert_called_once()

    def test_already_exited_ffmpeg_is_left_alone(self):
        bridge = build_bridge()
        proc = mock.Mock()
        proc.poll.return_value = 0
        bridge.process = proc
        bridge._terminate_ffmpeg()
        proc.terminate.assert_not_called()

    def test_orderly_shutdown_flag_defaults_off(self):
        self.assertFalse(build_bridge()._shutting_down)


class SilencePaddingTest(unittest.TestCase):
    def test_every_muxable_codec_has_a_silence_byte(self):
        # a reconnect pads the gap with silence, so each format we can mux needs one
        for fmt, _ in AUDIO_INPUTS.values():
            self.assertIn(fmt, AUDIO_SILENCE)

    def test_silence_bytes_match_the_encodings(self):
        self.assertEqual(AUDIO_SILENCE["alaw"], b"\xd5")
        self.assertEqual(AUDIO_SILENCE["mulaw"], b"\xff")

    def test_one_second_of_padding_is_one_second_of_samples(self):
        fmt, _ = AUDIO_INPUTS[1027]
        self.assertEqual(len(AUDIO_SILENCE[fmt] * int(1.0 * 16000)), 16000)

    def test_reconnect_delay_stays_short(self):
        # the delay becomes silence in the recording, and drops come every couple
        # of minutes, so a slow reconnect costs real audio
        self.assertLessEqual(AUDIO_RECONNECT_DELAY, 1.0)

    def test_giveup_is_long_enough_to_ride_out_a_drop(self):
        # the server drops these sockets every 40-90s; giving up sooner would put
        # us back to restarting the bridge, and video with it
        self.assertGreaterEqual(AUDIO_GIVEUP_SECONDS, 30.0)


class RateDetectionTest(unittest.TestCase):
    def test_measured_rates_snap_to_the_nearest_supported_one(self):
        # jitter and the odd dropped frame must not shift the answer
        for measured, expected in [
            (7900, 8000), (8200, 8000), (11000, 8000),
            (13000, 16000), (15994, 16000), (16400, 16000),
        ]:
            got = min(AUDIO_RATES, key=lambda r: abs(r - measured))
            self.assertEqual(got, expected, f"{measured} -> {got}")

    def test_burst_threshold_separates_buffered_frames_from_real_ones(self):
        # real frames are 40ms apart at 16kHz and 80ms at 8kHz; buffered ones ~1ms
        self.assertLess(AUDIO_BURST_GAP, 0.040)
        self.assertGreater(AUDIO_BURST_GAP, 0.002)

    def test_enough_frames_are_timed_for_a_median_to_mean_something(self):
        self.assertGreaterEqual(AUDIO_RATE_SAMPLE_FRAMES, 20)

    def test_median_survives_bursts_and_stalls(self):
        """The failure that shipped: a 2s window caught a quiet patch, measured
        276 bytes/s, snapped to 8 kHz and halved the session's audio speed."""

        def measure(gaps, size=640):
            kept = sorted(g for g in gaps if g >= AUDIO_BURST_GAP)
            return size / kept[len(kept) // 2]

        def snap(measured):
            return min(AUDIO_RATES, key=lambda r: abs(r - measured))

        cases = {
            "steady 16k": ([0.040] * 60, 16000),
            "steady 8k": ([0.080] * 60, 8000),
            "16k with a stall": ([0.040] * 50 + [3.0] * 10, 16000),
            # the burst is unbounded: skipping a fixed count once left every timed
            # frame still inside it, reading 744396 bytes/s
            "16k behind a long burst": ([0.001] * 200 + [0.040] * 50, 16000),
            "8k behind a long burst": ([0.001] * 500 + [0.080] * 50, 8000),
            "16k with burst and stall": (
                [0.001] * 100 + [0.040] * 40 + [3.0] * 10, 16000),
        }
        for name, (gaps, expected) in cases.items():
            self.assertEqual(snap(measure(gaps)), expected, name)

    def test_codec_table_no_longer_assumes_a_rate(self):
        # the codec id does not carry the rate; assuming 8 kHz played 16 kHz
        # cameras at half speed and starved the pipe
        for value in AUDIO_INPUTS.values():
            self.assertEqual(len(value), 2)


class TeardownOrderTest(unittest.TestCase):
    def test_ffmpeg_is_killed_before_the_audio_pipe_is_closed(self):
        # a writer parked in a blocking os.write only returns when the read end
        # goes away, and that write runs in a thread that cannot be cancelled
        bridge = build_bridge()
        order = []
        proc = mock.Mock()
        proc.poll.side_effect = [None, 0]
        proc.terminate.side_effect = lambda: order.append("kill_ffmpeg")
        bridge.process = proc
        bridge.audio_fd = 9
        with mock.patch("os.close", side_effect=lambda fd: order.append("close_pipe")):
            bridge._stop_ffmpeg()
        self.assertEqual(order, ["kill_ffmpeg", "close_pipe"])

    def test_stderr_read_is_bounded(self):
        # reading FFmpeg's stderr runs to EOF, which never arrives while it is
        # merely wedged, and it froze the event loop before teardown could run
        self.assertGreater(STDERR_READ_TIMEOUT, 0)
        self.assertLessEqual(STDERR_READ_TIMEOUT, 30)

    def test_shutdown_wait_is_bounded(self):
        self.assertGreater(SHUTDOWN_TIMEOUT, 0)
        self.assertLessEqual(SHUTDOWN_TIMEOUT, 30)


class CadencePlausibilityTest(unittest.TestCase):
    """A wrong rate is worse than no audio: it plays the whole session at the
    wrong speed, whereas video-only is merely a missing feature."""

    def decide(self, gaps, size=640):
        """Mirror of _measure_rate: only frames at a plausible cadence count, and
        sampling continues until there are enough of them or time runs out."""
        good = []
        elapsed = 0.0
        for gap in gaps:
            elapsed += gap
            if elapsed > AUDIO_RATE_MEASURE_TIMEOUT:
                break
            if AUDIO_GAP_MIN <= gap <= AUDIO_GAP_MAX:
                good.append(gap)
            if len(good) >= AUDIO_RATE_SAMPLE_FRAMES:
                break
        if len(good) < AUDIO_RATE_MIN_PLAUSIBLE:
            return None
        good.sort()
        return min(AUDIO_RATES,
                   key=lambda r: abs(r - size / good[len(good) // 2]))

    def test_settled_streams_are_measured(self):
        self.assertEqual(self.decide([0.040] * 60), 16000)
        self.assertEqual(self.decide([0.080] * 60), 8000)
        self.assertEqual(self.decide([0.001] * 300 + [0.040] * 60), 16000)

    def test_a_stream_that_never_settles_is_refused(self):
        # frames trickling at 144ms or 267ms against a steady 40ms read as 8kHz and
        # halved the session's speed, so they must not be measured
        self.assertIsNone(self.decide([0.267] * 200))
        self.assertIsNone(self.decide([0.144] * 200))

    def test_a_slow_start_delays_the_answer_rather_than_denying_it(self):
        # this shipped as "19 of 50 frames plausible" and dropped a whole session to
        # video-only, because sampling stopped before enough good frames arrived
        self.assertEqual(self.decide([0.267] * 31 + [0.040] * 200), 16000)
        self.assertEqual(self.decide([0.001] * 300 + [0.040] * 200), 16000)

    def test_a_ramp_that_settles_is_measured_from_the_settled_part(self):
        self.assertEqual(self.decide([0.267] * 20 + [0.040] * 60), 16000)

    def test_both_candidate_cadences_sit_inside_the_band(self):
        for gap in (0.040, 0.080):
            self.assertLessEqual(AUDIO_GAP_MIN, gap)
            self.assertGreaterEqual(AUDIO_GAP_MAX, gap)


class PacerTest(unittest.TestCase):
    """FFmpeg expects a steady stream and stalls the whole mux without one, which
    backs up into the video write and takes the bridge down. The websocket cannot
    promise steadiness, so the pacer supplies it."""

    RATE = 16000

    def run_pacer(self, supply, ticks=250):
        """Mirror of _pace_audio's arithmetic, with deliberately late wakeups."""
        queue = bytearray()
        written = silence = 0
        now = 0.0
        for i in range(ticks):
            now += AUDIO_PACE_INTERVAL * (1.6 if i % 37 == 0 else 1.0)
            queue.extend(supply(i))
            overflow = len(queue) - self.RATE
            if overflow > 0:
                del queue[:overflow]
            owed = int(now * self.RATE) - written
            if owed <= 0:
                continue
            have = min(owed, len(queue))
            del queue[:have]
            silence += owed - have
            written += owed
        return written, silence, now

    def assert_exact(self, supply):
        written, silence, now = self.run_pacer(supply)
        self.assertEqual(written, int(now * self.RATE))
        return silence

    def test_output_matches_the_declared_rate_whatever_arrives(self):
        # late wakeups are made up rather than accumulating into drift
        self.assert_exact(lambda i: b"x" * 320)
        self.assert_exact(lambda i: b"")
        self.assert_exact(lambda i: b"x" * 20000 if i == 0 else b"x" * 320)

    def test_a_dropout_becomes_silence_of_the_same_length(self):
        silence = self.assert_exact(lambda i: b"" if 50 < i < 110 else b"x" * 320)
        # 60 ticks of 20ms is about 1.2s
        self.assertAlmostEqual(silence / self.RATE, 1.2, delta=0.2)

    def test_silence_covers_the_whole_run_when_nothing_arrives(self):
        silence = self.assert_exact(lambda i: b"")
        self.assertGreater(silence, 0)

    def test_a_burst_is_spent_rather_than_dropped(self):
        self.assertEqual(
            self.assert_exact(lambda i: b"x" * 20000 if i == 0 else b"x" * 320), 0)


class QueueTrimTest(unittest.TestCase):
    RATE = 16000

    def run_queue(self, ticks=600, burst=20000):
        """Trimming to a target leaves headroom. Trimming to the limit does not, and
        the queue then sits pinned there with every frame tripping it — which shipped
        as 3540 trims in one session and left audio a second behind video."""
        top = int(self.RATE * AUDIO_QUEUE_MAX_SECONDS)
        target = int(self.RATE * AUDIO_QUEUE_TARGET_SECONDS)
        queue = bytearray()
        trims = 0
        for tick in range(ticks):
            queue.extend(b"x" * (burst if tick == 0 else 320))
            if len(queue) > top:
                del queue[: len(queue) - target]
                trims += 1
            take = min(320, len(queue))
            del queue[:take]
        return trims, len(queue) / self.RATE

    def test_a_burst_is_trimmed_once_not_every_frame(self):
        trims, _ = self.run_queue()
        self.assertEqual(trims, 1)

    def test_queue_settles_near_the_target(self):
        _, backlog = self.run_queue()
        self.assertLessEqual(backlog, AUDIO_QUEUE_TARGET_SECONDS + 0.05)

    def test_target_leaves_headroom_under_the_limit(self):
        self.assertLess(AUDIO_QUEUE_TARGET_SECONDS, AUDIO_QUEUE_MAX_SECONDS)

    def test_backlog_stays_short_enough_not_to_be_heard_as_lag(self):
        self.assertLessEqual(AUDIO_QUEUE_MAX_SECONDS, 1.0)


if __name__ == "__main__":
    unittest.main()
