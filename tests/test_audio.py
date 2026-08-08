"""Tests for audio codec selection and the FFmpeg command it produces."""
import os
import unittest
from unittest import mock

from micam import AUDIO_INPUTS, RTSPBridge


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
        cmd, kwargs = self.start(bridge, AUDIO_INPUTS[1027])
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
        cmd, _ = self.start(build_bridge(), AUDIO_INPUTS[1026])
        self.assertIn("mulaw", cmd)

    def test_video_input_precedes_audio_input(self):
        cmd, _ = self.start(build_bridge(), AUDIO_INPUTS[1027])
        # -map 0:v / -map 1:a rely on this order
        self.assertLess(cmd.index("pipe:0"), cmd.index("pipe:7"))

    def test_only_video_is_stamped_from_the_wallclock(self):
        cmd, _ = self.start(build_bridge(), AUDIO_INPUTS[1027])
        # stamping raw PCM by arrival time compresses bursts and runs audio ahead of video
        self.assertEqual(cmd.count("-use_wallclock_as_timestamps"), 1)
        self.assertLess(cmd.index("-use_wallclock_as_timestamps"), cmd.index("pipe:0"))


class AudioInputTableTest(unittest.TestCase):
    def test_opus_and_pcm_are_not_muxed_yet(self):
        self.assertNotIn(1032, AUDIO_INPUTS)
        self.assertNotIn(1024, AUDIO_INPUTS)

    def test_g711_pair_is_eight_kilohertz_mono(self):
        for codec_id in (1026, 1027):
            _, rate, channels = AUDIO_INPUTS[codec_id]
            self.assertEqual((rate, channels), (8000, 1))


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


if __name__ == "__main__":
    unittest.main()
