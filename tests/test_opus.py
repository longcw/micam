"""Tests for reading Opus packet headers and framing them as Ogg."""
import struct
import unittest

from micam.opus import OPUS_RATE, OggOpusWriter, packet_frame


def toc(config, stereo=False, code=0):
    return bytes([(config << 3) | (0x04 if stereo else 0) | code])


def pages(data):
    """Split a run of Ogg pages into (header_type, granule, seq, payload) tuples."""
    out = []
    i = 0
    while i < len(data):
        assert data[i:i + 4] == b"OggS", "page does not start with the capture pattern"
        header_type = data[i + 5]
        granule = struct.unpack("<q", data[i + 6:i + 14])[0]
        seq = struct.unpack("<I", data[i + 18:i + 22])[0]
        count = data[i + 26]
        segments = data[i + 27:i + 27 + count]
        body = i + 27 + count
        size = sum(segments)
        out.append((header_type, granule, seq, data[body:body + size]))
        i = body + size
    return out


class PacketFrameTest(unittest.TestCase):
    def test_silk_wideband_40ms_is_1920_samples(self):
        # what the 摄像机 4 4K sends: config 10, mono, one frame
        self.assertEqual(packet_frame(toc(10) + b"\x00" * 300), (1920, 1))

    def test_celt_fullband_20ms_is_960_samples(self):
        self.assertEqual(packet_frame(toc(31) + b"\x00"), (960, 1))

    def test_stereo_bit_is_read(self):
        self.assertEqual(packet_frame(toc(10, stereo=True) + b"\x00")[1], 2)

    def test_code_1_carries_two_frames(self):
        self.assertEqual(packet_frame(toc(10, code=1) + b"\x00")[0], 3840)

    def test_code_3_takes_its_frame_count_from_the_next_byte(self):
        self.assertEqual(packet_frame(toc(10, code=3) + bytes([3]))[0], 5760)


class OggOpusWriterTest(unittest.TestCase):
    def setUp(self):
        self.writer = OggOpusWriter()
        self.packet = toc(10) + b"\xab" * 300

    def test_first_packet_is_preceded_by_the_two_headers(self):
        out = pages(self.writer.wrap(self.packet))
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0][3][:8], b"OpusHead")
        self.assertEqual(out[1][3][:8], b"OpusTags")
        self.assertEqual(out[2][3], self.packet)

    def test_only_the_first_page_begins_the_stream(self):
        out = pages(self.writer.wrap(self.packet) + self.writer.wrap(self.packet))
        self.assertEqual(out[0][0], 0x02)
        self.assertEqual([p[0] for p in out[1:]], [0, 0, 0])

    def test_headers_are_written_once(self):
        self.writer.wrap(self.packet)
        out = pages(self.writer.wrap(self.packet))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], self.packet)

    def test_opus_head_announces_the_packet_channel_count(self):
        head = pages(self.writer.wrap(toc(10, stereo=True) + b"\x00"))[0][3]
        self.assertEqual(head[9], 2)
        # nothing re-encodes, so no samples are skipped at the front
        self.assertEqual(struct.unpack("<H", head[10:12])[0], 0)

    def test_granule_accumulates_across_packets(self):
        out = pages(self.writer.wrap(self.packet) + self.writer.wrap(self.packet))
        self.assertEqual([p[1] for p in out], [0, 0, 1920, 3840])

    def test_page_sequence_increments_without_gaps(self):
        out = pages(self.writer.wrap(self.packet) + self.writer.wrap(self.packet))
        self.assertEqual([p[2] for p in out], [0, 1, 2, 3])

    def test_long_packets_are_split_into_255_byte_segments(self):
        # a 601-byte packet: two full segments then a short one to terminate it
        out = pages(self.writer.wrap(toc(10) + b"\x00" * 600))
        self.assertEqual(out[2][3], toc(10) + b"\x00" * 600)

    def test_checksum_covers_the_page_with_its_own_field_zeroed(self):
        from micam.opus import _CRC_TABLE
        self.writer.wrap(self.packet)          # headers, so the next call is one page
        page = self.writer.wrap(self.packet)
        stated = struct.unpack("<I", page[22:26])[0]
        crc = 0
        for byte in page[:22] + b"\x00\x00\x00\x00" + page[26:len(page)]:
            crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ byte]
        self.assertEqual(stated, crc)

    def test_empty_packets_are_dropped(self):
        self.assertEqual(self.writer.wrap(b""), b"")


class SilenceTest(unittest.TestCase):
    def setUp(self):
        self.writer = OggOpusWriter()

    def test_nothing_to_fill_before_a_packet_has_arrived(self):
        # the gap is only meaningful once the stream's configuration is known
        self.assertEqual(self.writer.silence(2.0), b"")

    def test_gap_is_filled_with_frames_of_the_configuration_in_use(self):
        self.writer.wrap(toc(10) + b"\xab" * 300)
        out = pages(self.writer.silence(1.0))
        self.assertEqual(len(out), 25)  # 25 * 40 ms
        # a bare TOC is a zero-length frame, which decoders conceal as silence
        self.assertEqual(out[0][3], toc(10))

    def test_silence_keeps_the_granule_moving(self):
        self.writer.wrap(toc(10) + b"\xab" * 300)
        self.writer.silence(1.0)
        out = pages(self.writer.wrap(toc(10) + b"\xab" * 300))
        self.assertEqual(out[0][1], 1920 + OPUS_RATE + 1920)


if __name__ == "__main__":
    unittest.main()
