import os
import json
import asyncio
import aiohttp
import argparse
import logging
import subprocess
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MIoT codec ids, for logging cameras we cannot mux yet
AUDIO_CODEC_NAMES = {1024: 'PCM', 1026: 'G711U', 1027: 'G711A', 1032: 'OPUS'}

# codec id -> (ffmpeg input format, sample rate, channels). Cameras send bare frames with
# no container, so the format has to be declared rather than probed. Opus is absent on
# purpose: its frames need Ogg or RTP framing before FFmpeg will accept them.
AUDIO_INPUTS = {
    1026: ('mulaw', 8000, 1),
    1027: ('alaw', 8000, 1),
}

# how long to wait for the server to announce the codec before giving up on audio
AUDIO_CODEC_TIMEOUT = 10.0


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
            # no wallclock stamping here: raw PCM carries its own timing through the sample
            # rate, and stamping by arrival bunches bursts into too short a span
            ffmpeg_cmd += [
                '-f', audio_format,
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-i', f'pipe:{read_fd}',
            ]

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
        """Stop FFmpeg process."""
        self._close_audio_fd()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None

    async def _open_audio(self, session) -> Tuple[Optional[object], Optional[Tuple[str, int, int]]]:
        """Connect the audio stream and wait for the codec the camera is sending."""
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
                    logger.warning("No audio codec announced, continuing without audio")
                    break
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
                    logger.info("Audio codec %s(%s), muxing as %s", codec_name, codec_id, audio_input[0])
                    return ws, audio_input
                logger.warning(
                    "Audio codec %s(%s) is not supported yet, continuing without audio", codec_name, codec_id)
                break
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for the audio codec, continuing without audio")
        except Exception as e:
            logger.warning("Audio setup failed, continuing without it: %s", e)

        await ws.close()
        return None, None

    async def _stream_audio(self, ws):
        """Pipe audio frames into FFmpeg once video is flowing."""
        await self.video_started.wait()
        try:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=60.0)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await asyncio.wait_for(self.audio_write(msg.data), timeout=30.0)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    # a codec change mid-stream would need a new FFmpeg input, so stop instead of corrupting
                    logger.warning("Audio codec changed mid-stream, dropping audio: %s", msg.data)
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Audio WebSocket closed with error {ws.exception()}")
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    logger.info("Audio WebSocket connection close(%s)", msg.type)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Audio streaming stopped: %s", e)
        finally:
            self._close_audio_fd()
            # FFmpeg already announced an audio track to the RTSP server, and that
            # announcement cannot be withdrawn on a live session. Carrying on with
            # video would publish a stream advertising audio that never arrives,
            # which stalls consumers waiting on it. Restart the bridge instead, so
            # the next session either carries audio or honestly has none.
            if not self._shutting_down:
                logger.warning("Audio stream ended; restarting the bridge without a stale audio track")
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
                audio_ws, audio_input = await self._open_audio(session)

            self._start_ffmpeg(audio_input)
            audio_task = asyncio.create_task(self._stream_audio(audio_ws)) if audio_ws else None

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
                if audio_task:
                    audio_task.cancel()
                    await asyncio.gather(audio_task, return_exceptions=True)
                if audio_ws:
                    await audio_ws.close()
                self._stop_ffmpeg()
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
        if not self.process:
            raise RuntimeError("Process not started")
        if not self.process.stderr:
            return
        stderr = self.process.stderr.read().decode()
        if stderr:
            logger.error(f"FFmpeg stderr: %s", stderr)


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


if __name__ == "__main__":
    main()
