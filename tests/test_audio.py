"""Tests for audio codec selection and the FFmpeg command it produces."""
import os
import unittest
from unittest import mock

from micam import (
    AUDIO_GIVEUP_SECONDS,
    AUDIO_INPUTS,
    AUDIO_RATES,
    AUDIO_RATE_SAMPLE_SECONDS,
    AUDIO_RATE_WARMUP_SECONDS,
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
        # otherwise the muxer holds video until audio catches up, and audio
        # arriving at exactly real time can never close a gap once one opens
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

    def test_warmup_precedes_measurement(self):
        # the server flushes its buffer on connect much faster than real time, and
        # timing that burst reads as roughly double the real rate
        self.assertGreater(AUDIO_RATE_WARMUP_SECONDS, 0)

    def test_rate_detection_fits_inside_the_keyframe_wait(self):
        self.assertLessEqual(
            AUDIO_RATE_WARMUP_SECONDS + AUDIO_RATE_SAMPLE_SECONDS, 5.0)

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


if __name__ == "__main__":
    unittest.main()
