"""Bring back camera streams that stop carrying video, escalating from a camera refresh to a miloco restart."""
import os
import time
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# How often go2rtc is asked for its packet counters.
POLL_INTERVAL = 15.0

# No video for this long and miloco is asked to rebuild its camera managers. It costs
# nothing and touches no healthy stream, and it is all a failed startup refresh needs.
REFRESH_AFTER = 180.0

# Still no video this long and miloco is restarted. Every camera drops for about 2.5
# minutes while it comes back, so this waits well past the bridges' own 60 s retry.
RESTART_AFTER = 360.0

# A miloco younger than this is still starting, or was just restarted by its own
# policy after a crash; restarting it again only starts the wait over. A start that
# renews the spec cache takes about 5 minutes.
MILOCO_STARTUP_GRACE = 600.0

# At most one restart per window across all streams, so a camera that a restart cannot
# fix does not keep taking the healthy ones down with it.
RESTART_COOLDOWN = 1800.0


def video_packets(stream: dict) -> Dict[int, int]:
    """Map each video receiver of a go2rtc stream to its packet count."""
    return {
        receiver["id"]: receiver.get("packets", 0)
        for producer in stream.get("producers") or []
        for receiver in producer.get("receivers") or []
        if (receiver.get("codec") or {}).get("codec_type") == "video"
    }


class StreamWatchdog:
    def __init__(
        self,
        streams: List[str],
        refresh_after=REFRESH_AFTER,
        restart_after=RESTART_AFTER,
        restart_cooldown=RESTART_COOLDOWN,
    ):
        self.streams = streams
        self.refresh_after = refresh_after
        self.restart_after = restart_after
        self.restart_cooldown = restart_cooldown
        self._packets: Dict[str, Dict[int, int]] = {name: {} for name in streams}
        # when each stream last moved video. None until the first poll, so the
        # watchdog starting up is not mistaken for every stream having stalled
        self._last_video: Dict[str, Optional[float]] = {name: None for name in streams}
        # escalation already spent on the current stall, cleared when video returns
        self._refreshed: set = set()
        self._restarted: set = set()
        self._gave_up: set = set()
        self._last_restart: Optional[float] = None

    def observe(self, api_streams: dict, now: float, restart_allowed=True) -> List[str]:
        """Record one go2rtc snapshot and return the actions due: "refresh", "restart", or neither."""
        # record every stream's progress before acting, so a restart charged below
        # sees this round's state for the streams after it
        for name in self.streams:
            packets = video_packets(api_streams.get(name) or {})
            previous = self._packets[name]
            self._packets[name] = packets
            moved = any(count > previous.get(rid, 0) for rid, count in packets.items())
            if moved or self._last_video[name] is None:
                if name in self._refreshed or name in self._restarted:
                    logger.info("%s is carrying video again", name)
                self._last_video[name] = now
                self._refreshed.discard(name)
                self._restarted.discard(name)
                self._gave_up.discard(name)

        actions = []
        for name in self.streams:
            idle = now - self._last_video[name]
            if idle >= self.restart_after:
                if name in self._restarted:
                    if name not in self._gave_up:
                        logger.error(
                            "%s still has no video after a miloco restart; the camera itself "
                            "likely needs a power-cycle", name)
                        self._gave_up.add(name)
                elif restart_allowed and (
                        self._last_restart is None or now - self._last_restart >= self.restart_cooldown):
                    logger.warning("%s has had no video for %.0fs; restarting miloco", name, idle)
                    actions.append("restart")
                    self._last_restart = now
                    # every camera drops with miloco, so restart the clocks and charge the
                    # restart to each stream that was already stalled
                    for other in self.streams:
                        if now - self._last_video[other] >= self.refresh_after:
                            self._restarted.add(other)
                            self._refreshed.discard(other)
                        self._last_video[other] = now
                    break
            elif idle >= self.refresh_after and name not in self._refreshed:
                logger.warning("%s has had no video for %.0fs; refreshing miloco cameras", name, idle)
                self._refreshed.add(name)
                if "refresh" not in actions:
                    actions.append("refresh")
        return actions


async def refresh_cameras(session: aiohttp.ClientSession, base_url: str, username: str, password: str):
    """Log in to miloco and have it rebuild its camera managers."""
    payload = {"username": username, "password": password}
    async with session.post(f"{base_url}/api/auth/login", json=payload, ssl=False) as resp:
        resp.raise_for_status()
    async with session.get(f"{base_url}/api/miot/refresh_miot_cameras", ssl=False) as resp:
        logger.info("Camera refresh: %s %s", resp.status, await resp.text())


async def container_uptime(docker: aiohttp.ClientSession, container: str) -> float:
    """Seconds since the container last started, through the Docker Engine API."""
    async with docker.get(f"http://docker/containers/{container}/json") as resp:
        resp.raise_for_status()
        started_at = (await resp.json())["State"]["StartedAt"]
    # docker reports nanoseconds, which fromisoformat cannot parse
    started = datetime.fromisoformat(started_at[:19]).replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - started).total_seconds()


async def restart_container(docker: aiohttp.ClientSession, container: str):
    """Restart a container through the Docker Engine API."""
    # the restart waits for the stop, so allow for the container's grace period
    async with docker.post(
        f"http://docker/containers/{container}/restart",
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 204:
            raise RuntimeError(f"restart {container} failed: {resp.status} {await resp.text()}")
    logger.info("Restarted %s", container)


async def run(args):
    watchdog = StreamWatchdog(args.streams)
    logger.info("Watching %s on %s", ", ".join(args.streams), args.go2rtc_url)
    # miloco's cookie is set on a bare IP or a hosts entry, which the default jar refuses
    jar = aiohttp.CookieJar(unsafe=True)
    docker_connector = aiohttp.UnixConnector(path=args.docker_socket)
    async with aiohttp.ClientSession(cookie_jar=jar, timeout=aiohttp.ClientTimeout(total=30)) as session, \
            aiohttp.ClientSession(connector=docker_connector, timeout=aiohttp.ClientTimeout(total=30)) as docker:
        while True:
            try:
                async with session.get(f"{args.go2rtc_url}/api/streams") as resp:
                    api_streams = await resp.json()
            except Exception as e:
                # go2rtc down is not something a miloco restart fixes, so take no action
                logger.warning("Cannot read go2rtc streams: %r", e)
                api_streams = None

            try:
                restart_allowed = await container_uptime(docker, args.miloco_container) >= MILOCO_STARTUP_GRACE
            except Exception as e:
                logger.warning("Cannot inspect %s: %r", args.miloco_container, e)
                restart_allowed = False

            if api_streams is not None:
                for action in watchdog.observe(api_streams, time.monotonic(), restart_allowed):
                    try:
                        if action == "refresh":
                            await refresh_cameras(session, args.base_url, args.username, args.password)
                        else:
                            await restart_container(docker, args.miloco_container)
                    except Exception as e:
                        logger.error("%s failed: %r", action, e)

            await asyncio.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Recover camera streams that stop carrying video")
    parser.add_argument(
        "--streams", default=os.getenv("WATCH_STREAMS", ""), help="Comma separated go2rtc stream names")
    parser.add_argument("--go2rtc-url", default=os.getenv("GO2RTC_URL", "http://go2rtc:1984"))
    parser.add_argument("--base-url", default=os.getenv("MILOCO_BASE_URL", "https://miloco:8000"))
    parser.add_argument("--username", default=os.getenv("MILOCO_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("MILOCO_PASSWORD", ""))
    parser.add_argument("--miloco-container", default=os.getenv("MILOCO_CONTAINER", "miloco"))
    parser.add_argument("--docker-socket", default=os.getenv("DOCKER_SOCKET", "/var/run/docker.sock"))
    args = parser.parse_args()

    args.streams = [name.strip() for name in args.streams.split(",") if name.strip()]
    if not args.streams:
        logger.error("No streams to watch; set WATCH_STREAMS")
        return
    if not args.password:
        logger.error("Password is required")
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
