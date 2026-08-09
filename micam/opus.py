"""Framing bare Opus packets as Ogg, the only container FFmpeg reads them from.

Cameras send Opus packets with no container at all, and FFmpeg has no raw-Opus
demuxer, so each packet is wrapped in an Ogg page here.
"""
import struct
from typing import Optional, Tuple

# Opus always runs on a 48 kHz clock however the camera encoded it, so granule
# positions and the rate FFmpeg reports are this whatever the bandwidth.
OPUS_RATE = 48000

# Frame duration in 48 kHz samples for each of the 32 TOC configurations
# (RFC 6716 section 3.1): SILK at three bandwidths, hybrid at two, then CELT.
_SILK = (480, 960, 1920, 2880)  # 10, 20, 40, 60 ms
_HYBRID = (480, 960)            # 10, 20 ms
_CELT = (120, 240, 480, 960)    # 2.5, 5, 10, 20 ms
_FRAME_SAMPLES = _SILK * 3 + _HYBRID * 2 + _CELT * 4

# Ogg checksums with polynomial 0x04c11db7, no reflection and no final xor, which
# is neither zlib.crc32 nor binascii.crc32 — hence the table.
_CRC_TABLE = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if _r & 0x80000000 else (_r << 1) & 0xFFFFFFFF
    _CRC_TABLE.append(_r)


def packet_frame(packet: bytes) -> Tuple[int, int]:
    """Samples at 48 kHz and channel count for one Opus packet, read from its TOC byte."""
    toc = packet[0]
    per_frame = _FRAME_SAMPLES[toc >> 3]
    channels = 2 if toc & 0x04 else 1
    code = toc & 0x03
    if code == 0:
        frames = 1
    elif code < 3:
        frames = 2
    else:
        # code 3 counts its frames in the six low bits of the byte after the TOC
        frames = packet[1] & 0x3F
    return per_frame * frames, channels


class OggOpusWriter:
    """Turns Opus packets into an Ogg bitstream, one page per packet.

    One instance per FFmpeg process: the serial, page sequence and granule
    position have to run unbroken for the life of the stream, however many times
    the camera's audio socket drops and reopens underneath it.
    """

    def __init__(self, serial: int = 0x6D6963):
        self.serial = serial
        self._seq = 0
        self._granule = 0
        # first packet's TOC, kept so a gap can be filled with frames of the same
        # configuration; None until a packet has actually arrived
        self._silence: Optional[bytes] = None

    def wrap(self, packet: bytes) -> bytes:
        """Bytes to write for one Opus packet, preceded by the headers on the first call."""
        if not packet:
            return b""
        samples, channels = packet_frame(packet)
        prefix = b""
        if self._silence is None:
            # same configuration, one frame, no data: decoders treat a zero-length
            # frame as a loss and conceal it with silence
            self._silence = bytes([packet[0] & 0xFC])
            # pre-skip is 0 because nothing here re-encodes: every sample the camera
            # sent is one a listener should hear. The input rate that follows is
            # informational only, and channel mapping family 0 covers mono and stereo.
            head = (b"OpusHead" + bytes([1, channels]) + struct.pack("<H", 0)
                    + struct.pack("<I", OPUS_RATE) + struct.pack("<h", 0) + bytes([0]))
            tags = b"OpusTags" + struct.pack("<I", 5) + b"micam" + struct.pack("<I", 0)
            prefix = self._page(head, 0, header_type=0x02) + self._page(tags, 0)
        self._granule += samples
        return prefix + self._page(packet, self._granule)

    def silence(self, seconds: float) -> bytes:
        """Pages covering a gap, so later packets keep their place on the timeline."""
        if self._silence is None:
            return b""
        per_frame, _ = packet_frame(self._silence)
        pages = []
        for _ in range(int(seconds * OPUS_RATE / per_frame)):
            self._granule += per_frame
            pages.append(self._page(self._silence, self._granule))
        return b"".join(pages)

    def _page(self, packet: bytes, granule: int, header_type: int = 0) -> bytes:
        # lacing splits a packet into 255-byte segments, the last one short; an Opus
        # packet never approaches the 255 segments a single page holds
        segments = [255] * (len(packet) // 255) + [len(packet) % 255]
        page = (b"OggS" + bytes([0, header_type])
                + struct.pack("<q", granule)
                + struct.pack("<I", self.serial)
                + struct.pack("<I", self._seq)
                + b"\x00\x00\x00\x00"  # checksum, computed over the page with this zeroed
                + bytes([len(segments)]) + bytes(segments)
                + packet)
        self._seq += 1
        crc = 0
        for byte in page:
            crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ byte]
        return page[:22] + struct.pack("<I", crc) + page[26:]
