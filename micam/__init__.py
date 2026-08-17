import os
import json
import asyncio
import aiohttp
import argparse
import logging
import subprocess
from typing import Optional, Tuple

from .opus import OPUS_RATE, OggOpusWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MIoT codec ids, for logging cameras we cannot mux yet
AUDIO_CODEC_NAMES = {1024: 'PCM', 1026: 'G711U', 1027: 'G711A', 1032: 'OPUS'}

# codec id -> (ffmpeg input format, channels). Cameras send bare frames with no
# container, so the format has to be declared rather than probed. G.711's sample rate
# is measured instead of listed here, because the codec id does not carry it and
# cameras run it at either 8 or 16 kHz. Opus frames are wrapped in Ogg on the way to
# FFmpeg, which declares both rate and channels itself, so neither is listed.
AUDIO_INPUTS = {
    1026: ('mulaw', 1),
    1027: ('alaw', 1),
    1032: ('ogg', 0),
}

# Sample rates G.711 cameras use. The measured byte rate is snapped to the nearest,
# so a little jitter or a dropped frame cannot shift the answer.
AUDIO_RATES = (8000, 16000)

# Frames arriving closer together than this came from the server's buffer rather than
# the camera, and are ignored. Real frames are 40 ms apart at 16 kHz and 80 ms at
# 8 kHz, while buffered ones arrive in about a millisecond, so the two are far apart
# and the exact threshold does not matter.
AUDIO_BURST_GAP = 0.005

# The cadence a settled G.711 camera keeps. 640-byte frames land every 40 ms at
# 16 kHz and every 80 ms at 8 kHz, so anything outside this band is the stream still
# ramping up rather than the camera's pace, and timing it picks the wrong rate.
AUDIO_GAP_MIN = 0.015
AUDIO_GAP_MAX = 0.120

# How many frames must arrive at a sensible cadence before the reading is trusted.
AUDIO_RATE_MIN_PLAUSIBLE = 20

# Frames to time once the burst is out of the way, and the longest we will spend
# gathering them before going with what we have. At 25 frames a second the sample is
# about two seconds.
AUDIO_RATE_SAMPLE_FRAMES = 50
AUDIO_RATE_MEASURE_TIMEOUT = 15.0

# How far the measurement may sit from the rate we pick before it is worth warning
# about. Anything larger means the stream is not behaving like either candidate.
AUDIO_RATE_MAX_DRIFT = 0.35

# how long to wait for the server to announce the codec before giving up on audio
AUDIO_CODEC_TIMEOUT = 10.0

# Byte that encodes silence, per ffmpeg input format. Written to cover a gap so
# the audio clock keeps pace with wall clock across a reconnect.
AUDIO_SILENCE = {"alaw": b"\xd5", "mulaw": b"\xff"}

# Pause before reopening a dropped audio socket. Every second spent here becomes
# silence in the recording, and the server drops the socket often enough for that
# to add up, so keep it short; the give-up timer below is what stops a genuinely
# dead server from being hammered.
AUDIO_RECONNECT_DELAY = 0.5

# A write blocking is FFmpeg holding audio back until video catches up, which is
# normal and brief. This only catches FFmpeg having stopped reading altogether.
AUDIO_WRITE_TIMEOUT = 30.0

# Longest we wait for the audio task to stop during teardown. Exceeding it means a
# worker thread is still stuck, and the bridge is better off exiting so the restart
# policy can bring it back than hanging on forever.
SHUTDOWN_TIMEOUT = 10.0

# Longest we wait for FFmpeg's stderr to reach EOF after asking it to exit.
STDERR_READ_TIMEOUT = 5.0

# How long audio may stay down before the bridge restarts rather than keep
# advertising a track it cannot deliver.
AUDIO_GIVEUP_SECONDS = 60.0


class RTSPBridge:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        camera_id: str,
        rtsp_url: str,
        video_codec='hevc',
        channel=0,
        enable_audio=True,
    ):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.camera_id = camera_id
        self.channel = channel
        self.video_codec = video_codec
        self.rtsp_url = rtsp_url
        self.enable_audio = enable_audio
        self.process: Optional[subprocess.Popen] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.waiting_for_keyframe = True
        self.audio_fd: Optional[int] = None
        # audio is held back until video starts, so both inputs share a wallclock origin
        self.video_started = asyncio.Event()
        # set once the video loop is tearing down, so a shutting-down audio task
        # does not mistake an orderly stop for an audio failure
        self._shutting_down = False
        # silence byte and sample rate for the running session, and when audio went
        # away, so a reconnect can fill the gap it left
        self._audio_silence: Optional[bytes] = None
        self._audio_rate: Optional[int] = None
        self._audio_gap_start: Optional[float] = None
        # set for the life of an Opus session, which is framed as Ogg on the way out
        self._ogg: Optional[OggOpusWriter] = None

    async def _login(self) -> bool:
        """Login and retrieve access token."""
        login_url = f"{self.base_url}/api/auth/login"
        payload = {"username": self.username, "password": self.password}

        try:
            async with self.session.post(login_url, json=payload, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Login successful. %s", data)

                    # Call login_status to ensure session is active
                    status_url = f"{self.base_url}/api/miot/login_status"
                    async with self.session.get(status_url, ssl=False) as status_resp:
                        if status_resp.status == 200:
                            return True
                        else:
                            logger.error(f"Login status check failed: {status_resp.status}")
                            return False
                else:
                    logger.error(f"Login failed: {response.status} - {await response.text()}")
                    return False
        except Exception as e:
            logger.error(f"Login exception: {e}")
            await asyncio.sleep(3)
            return False

    def _ws_url(self, kind: str) -> str:
        protocol = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://")[1]
        return f"{protocol}://{host}/api/miot/ws/{kind}?camera_id={self.camera_id}&channel={self.channel}"

    def _start_ffmpeg(self, audio_input: Optional[Tuple[str, int, int]] = None):
        """Start FFmpeg process, muxing audio from a second pipe when one is available."""
        # FFmpeg command to read from stdin and publish to RTSP
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-v', 'error',
            '-hide_banner',
            '-use_wallclock_as_timestamps', '1',  # Generate timestamps from arrival time
            '-analyzeduration', '20000000',  # 20 seconds
            '-probesize', '20000000',  # 20 MB
            '-f', self.video_codec,  # Input format
            '-i', 'pipe:0',  # Read from stdin
        ]

        pass_fds = ()
        if audio_input:
            audio_format, sample_rate, channels = audio_input
            read_fd, write_fd = os.pipe()
            self.audio_fd = write_fd
            pass_fds = (read_fd,)
            # no wallclock stamping here: the audio carries its own timing, through the
            # sample rate for raw PCM and granule positions for Ogg, and stamping by
            # arrival bunches bursts into too short a span
            ffmpeg_cmd += ['-f', audio_format]
            if audio_format != 'ogg':
                # Ogg announces rate and channels in its own header; raw PCM cannot.
                ffmpeg_cmd += ['-ar', str(sample_rate), '-ac', str(channels)]
            ffmpeg_cmd += ['-i', f'pipe:{read_fd}']

        if audio_input:
            # Write packets as they arrive rather than buffering to interleave them.
            # Precautionary: a live stream has nothing to gain from the muxer holding
            # video back for audio. Kept as cheap insurance, not because it was shown
            # to fix anything.
            ffmpeg_cmd += ['-max_interleave_delta', '0']

        ffmpeg_cmd += ['-map', '0:v', '-c:v', 'copy']  # Copy video stream
        if audio_input:
            ffmpeg_cmd += ['-map', '1:a', '-c:a', 'copy']  # Copy audio stream
        ffmpeg_cmd += [
            '-f', 'rtsp',  # Output format
            '-rtsp_transport', 'tcp',  # Use TCP for RTSP
            self.rtsp_url,
        ]

        logger.info(f"Starting FFmpeg: {' '.join(ffmpeg_cmd)}")
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # Suppress FFmpeg stdout
            stderr=subprocess.PIPE,  # Capture stderr for debugging if needed
            pass_fds=pass_fds,
        )
        for fd in pass_fds:
            os.close(fd)

    def _close_audio_fd(self):
        """Close the audio pipe so FFmpeg sees EOF rather than waiting on an input that stopped."""
        if self.audio_fd is None:
            return
        try:
            os.close(self.audio_fd)
        except OSError:
            pass
        self.audio_fd = None

    def _terminate_ffmpeg(self):
        """Signal FFmpeg to exit, leaving the process handle for the caller to reap."""
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            pass

    def _stop_ffmpeg(self):
        """Stop FFmpeg, then release the audio pipe.

        FFmpeg goes first on purpose. A writer parked in a blocking ``os.write``
        on a full pipe only comes back when the read end goes away, and that
        writer runs in a worker thread, which cannot be cancelled — closing our
        own end would leave it stuck.
        """
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.error("FFmpeg ignored SIGKILL")
            self.process = None
        self._close_audio_fd()

    async def _measure_rate(self, ws) -> Optional[int]:
        """Work out the sample rate by timing the frames the camera actually sends.

        Nothing announces it: the codec id says G.711 but not whether the camera
        runs it at narrowband or wideband, and cameras differ. G.711 carries one
        byte per sample, so the arriving byte rate is the sample rate. Guessing
        wrong is not a subtle error — assuming 8 kHz for a 16 kHz camera plays
        every recording at half speed, and starves FFmpeg's pipe until the socket
        dies, because it drains half as fast as the camera fills it.
        """
        loop = asyncio.get_running_loop()
        prev: Optional[float] = None
        gaps: list[float] = []
        sizes: list[int] = []
        give_up = loop.time() + AUDIO_RATE_MEASURE_TIMEOUT
        try:
            while len(gaps) < AUDIO_RATE_SAMPLE_FRAMES:
                if loop.time() > give_up:
                    break
                msg = await asyncio.wait_for(ws.receive(), timeout=AUDIO_CODEC_TIMEOUT)
                if msg.type != aiohttp.WSMsgType.BINARY:
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning("Audio WebSocket closed while measuring its rate")
                        return None
                    continue
                now = loop.time()
                gap = now - prev if prev is not None else None
                prev = now
                # Only frames arriving at a cadence one of the candidates would keep
                # count towards the sample. Buffered ones arrive back to back and a
                # reconnecting stream trickles, and neither says anything about the
                # camera's pace. Waiting for enough good frames rather than taking
                # whatever the first fifty happen to be means a slow start delays the
                # answer instead of denying it.
                if gap is None or not (AUDIO_GAP_MIN <= gap <= AUDIO_GAP_MAX):
                    continue
                gaps.append(gap)
                sizes.append(len(msg.data))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Could not measure the audio rate: %s", e)
            return None

        if len(gaps) < AUDIO_RATE_MIN_PLAUSIBLE:
            logger.warning(
                "Audio never settled into a steady cadence (%d usable frames in %.0fs); "
                "publishing video only this session",
                len(gaps), AUDIO_RATE_MEASURE_TIMEOUT)
            return None
        # Take the typical gap between frames rather than an average over a window.
        # A window is at the mercy of whatever happens to fall inside it: one quiet
        # patch once read as 276 bytes/s, which snapped to 8 kHz and played that whole
        # session at half speed. Bursts and stalls both sit in the tails, so the
        # median ignores them.
        gaps.sort()
        sizes.sort()
        typical_gap = gaps[len(gaps) // 2]
        typical_size = sizes[len(sizes) // 2]
        if typical_gap <= 0:
            logger.warning("Audio frames carried no usable timing")
            return None
        measured = typical_size / typical_gap
        rate = min(AUDIO_RATES, key=lambda r: abs(r - measured))
        drift = abs(measured - rate) / rate
        log = logger.warning if drift > AUDIO_RATE_MAX_DRIFT else logger.info
        log("Audio measures %.0f bytes/s (%d B every %.0f ms), treating it as %d Hz",
            measured, typical_size, typical_gap * 1000, rate)
        return rate

    async def _open_audio(
        self, session, known_rate: Optional[int] = None, hold_on_timeout: bool = False
    ) -> Tuple[Optional[object], Optional[Tuple[str, int, int]]]:
        """Connect the audio stream and wait for the codec the camera is sending.

        ``known_rate`` skips the measurement on a reconnect: FFmpeg is already
        running with that rate on its input and cannot be retuned mid-session, so
        measuring again would only add silence to the gap.

        ``hold_on_timeout`` returns the socket with no codec rather than closing
        it, leaving the caller free to keep listening.
        """
        ws_url = self._ws_url("audio_stream")
        logger.info(f"Connecting to audio WebSocket: {ws_url}")
        try:
            ws = await session.ws_connect(ws_url, ssl=False)
        except Exception as e:
            logger.warning("Audio unavailable, continuing without it: %s", e)
            return None, None

        try:
            deadline = asyncio.get_running_loop().time() + AUDIO_CODEC_TIMEOUT
            while True:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    raise asyncio.TimeoutError
                msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning("Audio WebSocket closed before announcing a codec")
                        break
                    continue
                codec_id = json.loads(msg.data).get("codec_id")
                codec_name = AUDIO_CODEC_NAMES.get(codec_id, "unknown")
                audio_input = AUDIO_INPUTS.get(codec_id)
                if audio_input:
                    fmt, channels = audio_input
                    if fmt == 'ogg':
                        # nothing to measure: Opus states its own frame duration packet
                        # by packet, and always decodes on a 48 kHz clock
                        rate = OPUS_RATE
                    else:
                        rate = known_rate or await self._measure_rate(ws)
                    if rate is None:
                        break
                    logger.info(
                        "Audio codec %s(%s) at %d Hz, muxing as %s", codec_name, codec_id, rate, fmt)
                    return ws, (fmt, rate, channels)
                logger.warning(
                    "Audio codec %s(%s) is not supported yet, continuing without audio", codec_name, codec_id)
                break
        except asyncio.TimeoutError:
            # The codec comes off the camera's first audio frame, and a cold stream
            # can take minutes to send one, so a longer wait here would only delay
            # video. Hold the socket instead and let the caller act when it arrives.
            if hold_on_timeout:
                logger.warning("No audio codec yet; publishing video-only and still listening")
                return ws, None
            logger.warning("Timed out waiting for the audio codec, continuing without audio")
        except Exception as e:
            logger.warning("Audio setup failed, continuing without it: %s", e)

        await ws.close()
        return None, None

    async def _watch_late_audio(self, ws) -> None:
        """Restart the bridge if the codec turns up once video-only is running.

        FFmpeg's inputs are fixed for the life of the process, so audio cannot join
        this session. The next one gets it announced immediately, because by then
        the camera is sending frames and the server has the codec cached.
        """
        try:
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    codec_id = json.loads(msg.data).get("codec_id")
                    codec_name = AUDIO_CODEC_NAMES.get(codec_id, "unknown")
                    if codec_id in AUDIO_INPUTS:
                        logger.error("Audio codec %s(%s) arrived late; restarting the bridge",
                                     codec_name, codec_id)
                        self._terminate_ffmpeg()
                    else:
                        logger.warning("Late audio codec %s(%s) is not supported, staying video-only",
                                       codec_name, codec_id)
                    return
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("Audio WebSocket closed before announcing a codec")
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Gave up waiting for a late audio codec: %s", e)

    async def _drain_audio(self, ws) -> None:
        """Pipe frames from one audio websocket into FFmpeg until it ends.

        Frames go straight through, and the write is allowed to block. FFmpeg keeps
        its inputs aligned with each other, so when audio gets ahead of video it stops
        reading this pipe until video catches up — the block is that signal, not a
        fault. Writing on our own schedule instead overrode it, and audio outran video
        every time the camera's frames arrived a little late.
        """
        live = False
        prev: Optional[float] = None
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=60.0)
            if msg.type == aiohttp.WSMsgType.BINARY:
                now = asyncio.get_running_loop().time()
                gap = now - prev if prev is not None else None
                prev = now
                if not live:
                    # The server flushes its buffer on connect, back to back. Writing
                    # that would start audio ahead of video, which is the very state
                    # FFmpeg then stalls to correct, so wait for live pace first.
                    if gap is None or gap < AUDIO_BURST_GAP:
                        continue
                    live = True
                    if self._audio_gap_start is not None:
                        await self._fill_gap(now - self._audio_gap_start)
                        self._audio_gap_start = None
                data = self._ogg.wrap(msg.data) if self._ogg else msg.data
                await asyncio.wait_for(self.audio_write(data), timeout=AUDIO_WRITE_TIMEOUT)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                # a codec change would need a different FFmpeg input, which a live
                # session cannot be given, so stop rather than emit garbled audio
                logger.warning("Audio codec changed mid-stream: %s", msg.data)
                raise RuntimeError("audio codec changed")
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"audio websocket error: {ws.exception()}")
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                raise RuntimeError(f"audio websocket closed ({msg.type})")

    async def _fill_gap(self, seconds: float) -> None:
        """Write silence for an audio gap so later samples keep their place in time.

        FFmpeg times this input by counting samples, so skipping a gap shifts every
        later sample earlier and slides audio ahead of video for good.
        """
        seconds = min(seconds, AUDIO_GIVEUP_SECONDS)
        if seconds <= 0:
            return
        if self._ogg:
            filler = self._ogg.silence(seconds)
        elif self._audio_silence and self._audio_rate:
            filler = self._audio_silence * int(seconds * self._audio_rate)
        else:
            filler = b""
        if not filler:
            return
        logger.info("Audio resumed; filling %.1fs of silence", seconds)
        await asyncio.wait_for(self.audio_write(filler), timeout=AUDIO_WRITE_TIMEOUT)

    async def _stream_audio(self, session, ws, audio_input):
        """Feed FFmpeg audio, reconnecting as needed, while video keeps running.

        The server drops these sockets periodically. Tearing the whole bridge down each
        time also killed a perfectly healthy video stream, so the socket is reopened
        and the gap filled with silence. Only audio that stays away for
        AUDIO_GIVEUP_SECONDS restarts the bridge, since by then the stream is
        advertising a track it can no longer deliver.
        """
        await self.video_started.wait()
        fmt, rate, _ = audio_input
        self._ogg = OggOpusWriter() if fmt == 'ogg' else None
        self._audio_silence = AUDIO_SILENCE.get(fmt, b"\xff")
        self._audio_rate = rate
        self._audio_gap_start = None
        down_since: Optional[float] = None
        try:
            while not self._shutting_down:
                try:
                    await self._drain_audio(ws)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("Audio interrupted: %s", e)
                finally:
                    if self._audio_gap_start is None:
                        self._audio_gap_start = asyncio.get_running_loop().time()
                    if ws is not None and not ws.closed:
                        await ws.close()
                    ws = None

                now = asyncio.get_running_loop().time()
                if down_since is None:
                    down_since = now
                if now - down_since > AUDIO_GIVEUP_SECONDS:
                    logger.error(
                        "Audio has been unavailable for %.0fs; restarting the bridge",
                        now - down_since)
                    return
                await asyncio.sleep(AUDIO_RECONNECT_DELAY)
                if self._shutting_down:
                    return
                ws, reconnected = await self._open_audio(session, known_rate=rate)
                if ws is None:
                    continue
                if reconnected != audio_input:
                    # FFmpeg's audio input is fixed for the life of the process, so a
                    # camera that came back speaking differently needs a fresh one
                    logger.warning("Audio came back as %s, not %s; restarting the bridge",
                                   reconnected, audio_input)
                    await ws.close()
                    return
                logger.info("Audio resumed after %.1fs", asyncio.get_running_loop().time() - down_since)
                down_since = None
        except asyncio.CancelledError:
            raise
        finally:
            if ws is not None and not ws.closed:
                await ws.close()
            self._close_audio_fd()
            # FFmpeg already announced an audio track to the RTSP server, and that
            # announcement cannot be withdrawn on a live session. Publishing a
            # stream that advertises audio it never sends stalls consumers waiting
            # on it, so give the bridge a clean restart instead.
            if not self._shutting_down:
                logger.warning("Audio gone for good; restarting the bridge")
                self._terminate_ffmpeg()

    async def run(self):
        """Main loop to connect to WebSocket and pipe data."""
        jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(cookie_jar=jar) as session:
            self.session = session
            if not await self._login():
                return

            audio_ws = None
            audio_input = None
            if self.enable_audio:
                audio_ws, audio_input = await self._open_audio(session, hold_on_timeout=True)

            self._start_ffmpeg(audio_input)
            if audio_input:
                audio_task = asyncio.create_task(
                    self._stream_audio(session, audio_ws, audio_input))
            elif audio_ws:
                audio_task = asyncio.create_task(self._watch_late_audio(audio_ws))
            else:
                audio_task = None

            ws_url = self._ws_url("video_stream")
            logger.info(f"Connecting to WebSocket: {ws_url}")

            try:
                async with session.ws_connect(ws_url, ssl=False) as ws:
                    logger.info("WebSocket connected. Streaming data...")

                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=60.0)
                        except asyncio.TimeoutError:
                            logger.error("Data received timeout. Exiting.")
                            break

                        if msg.type == aiohttp.WSMsgType.BINARY:
                            try:
                                data_len = len(msg.data)
                                if data_len >= 100:
                                    logger.debug("Received binary data: %s", data_len)

                                if self.waiting_for_keyframe:
                                    if self._is_keyframe(msg.data):
                                        logger.info("Keyframe detected! Starting stream...")
                                        self.waiting_for_keyframe = False
                                        self.video_started.set()
                                    else:
                                        logger.debug("Skipping non-keyframe data...")
                                        continue
                                await asyncio.wait_for(self.process_write(msg.data), timeout=30.0)
                            except asyncio.TimeoutError:
                                logger.error("Write data to process timeout.")
                                await self.process_stderr()
                                break
                            except BrokenPipeError as e:
                                logger.error("FFmpeg process terminated unexpectedly. %s", e)
                                await self.process_stderr()
                                break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket connection closed with error {ws.exception()}")
                            break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                            logger.info("WebSocket connection close(%s)", msg.type)
                            break
                        else:
                            logger.info(f"Unexpected WebSocket message type: {msg.type}")
            except Exception as e:
                logger.error(f"Streaming error", exc_info=True)
            finally:
                self._shutting_down = True
                self.video_started.set()
                # Tear FFmpeg down before waiting on the audio task. That task can
                # be parked in a blocking write on a full pipe, which no amount of
                # cancelling will interrupt; killing the reader is what frees it.
                # Waiting first hung the process here, so it never exited and the
                # restart policy never fired — the stream stayed dead until a
                # manual restart.
                self._stop_ffmpeg()
                if audio_task:
                    audio_task.cancel()
                    done, pending = await asyncio.wait(
                        {audio_task}, timeout=SHUTDOWN_TIMEOUT)
                    if pending:
                        logger.error("Audio task did not stop; exiting anyway")
                if audio_ws and not audio_ws.closed:
                    await audio_ws.close()
                logger.info("Stream finished")

    def _is_keyframe(self, data: bytes) -> bool:
        if self.video_codec == 'h264':
            i = 0
            while i < len(data) - 4:
                if (
                    data[i] == 0x00 and data[i + 1] == 0x00 and
                    ((data[i + 2] == 0x00 and data[i + 3] == 0x01) or data[i + 2] == 0x01)
                ):
                    nal_unit_type = (data[i + 3] & 0x1f) if data[i + 2] == 0x01 else (data[i + 4] & 0x1f)
                    return nal_unit_type == 5
                i += 1
            return False
        elif self.video_codec == 'hevc':
            i = 0
            while i < len(data) - 6:
                if (
                    data[i] == 0x00 and data[i + 1] == 0x00 and
                    ((data[i + 2] == 0x00 and data[i + 3] == 0x01) or data[i + 2] == 0x01)
                ):
                    nal_start = i + 3 if data[i + 2] == 0x01 else i + 4
                    nal_unit_type = (data[nal_start] >> 1) & 0x3f
                    if nal_unit_type in [16, 17, 18, 19, 20]:
                        return True
                i += 1
            return False
        return True

    async def process_write(self, data):
        if not self.process:
            raise RuntimeError("Process not started")
        if not self.process.stdin:
            raise RuntimeError("Process has no stdin")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._process_write, data)

    def _process_write(self, data):
        self.process.stdin.write(data)
        self.process.stdin.flush()

    async def audio_write(self, data):
        if self.audio_fd is None:
            raise RuntimeError("Audio pipe not started")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._audio_write, data)

    def _audio_write(self, data):
        os.write(self.audio_fd, data)

    async def process_stderr(self):
        """Log whatever FFmpeg wrote to stderr, without waiting on a live process.

        The read runs to EOF, and EOF only arrives once FFmpeg exits, so calling
        this while it is merely wedged blocks forever — and on the event loop,
        which freezes the whole bridge before it can tear anything down. Send it
        on its way first, then read off the loop with a deadline.
        """
        proc = self.process
        if proc is None or proc.stderr is None:
            return
        self._terminate_ffmpeg()
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, proc.stderr.read), STDERR_READ_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Gave up reading FFmpeg stderr")
            return
        except Exception as e:
            logger.error("Could not read FFmpeg stderr: %s", e)
            return
        text = data.decode(errors="replace") if data else ""
        if text.strip():
            logger.error("FFmpeg stderr: %s", text.strip())


def main():
    parser = argparse.ArgumentParser(description="Bridge WebSocket video stream to RTSP")
    parser.add_argument("--base-url", default="", help="Base URL of the Miloco server")
    parser.add_argument("--username", default="admin", help="Login username")
    parser.add_argument("--password", default="", help="Login password (MD5)")
    parser.add_argument("--camera-id", default="", help="Camera ID to stream")
    parser.add_argument("--channel", default="", help="Camera channel")
    parser.add_argument("--video-codec", default="hevc", help="Input video codec (hevc or h264)")
    parser.add_argument("--rtsp-url", default="", help="Target RTSP URL")
    parser.add_argument(
        "--no-audio", action="store_true", help="Publish video only, even if the camera sends audio")

    args = parser.parse_args()

    password = args.password or os.getenv("MILOCO_PASSWORD", "")
    if not password:
        logger.error("Password is required")
        return

    camera_id = args.camera_id or os.getenv("CAMERA_ID", "")
    if not camera_id:
        logger.error("Camera ID is required")
        return

    enable_audio = not args.no_audio and os.getenv("ENABLE_AUDIO", "1").lower() not in ("0", "false", "no")

    bridge = RTSPBridge(
        base_url=args.base_url or os.getenv("MILOCO_BASE_URL", "https://miloco:8000"),
        username=args.username or os.getenv("MILOCO_USERNAME", "admin"),
        password=password,
        camera_id=camera_id,
        rtsp_url=args.rtsp_url or os.getenv("RTSP_URL", "rtsp://0.0.0.0:8554/live"),
        video_codec=args.video_codec or os.getenv("VIDEO_CODEC", "hevc"),
        channel=args.channel or os.getenv("STREAM_CHANNEL", "0"),
        enable_audio=enable_audio,
    )

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        # A worker thread still parked in a blocking write would keep the
        # interpreter alive through executor shutdown, so the process would never
        # exit and the restart policy would never bring the stream back. There is
        # no state worth unwinding here, so leave immediately.
        logging.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
