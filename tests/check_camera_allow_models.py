"""Check CAMERA_ALLOW_MODELS against the real camera_extra_info.yaml.

Runs inside a miloco container carrying the patched camera.py, because the lists it
overrides ship in the image rather than this repo:

    docker exec -i -w /app <container> python - < tests/check_camera_allow_models.py
"""
import asyncio
import os
import sys

sys.path.insert(0, "/app")


def passes(info, model):
    """Mirror of the model filter in client.get_cameras_async."""
    device_class = model.split(".")[1]
    if device_class not in info.allow_classes:
        return False
    if device_class in info.denylist:
        return model not in info.denylist[device_class]
    if device_class in info.allowlist:
        return model in info.allowlist[device_class]
    return False


async def load(allow_models):
    import miot.camera as cam
    os.environ["CAMERA_ALLOW_MODELS"] = allow_models
    await cam.get_camera_extra_info.cache.clear()
    return await cam.get_camera_extra_info()


# (model, allowed without the override, allowed once the override names it)
CASES = [
    ("chuangmi.camera.021a04", False, True),   # denylisted class, entry removed
    ("xiaomi.wifispeaker.zzz9", False, True),  # allowlisted class, entry added
    ("acme.doorbell.x1", False, True),         # class in neither list
]
# these must not move whatever the override says
UNTOUCHED = [
    ("chuangmi.camera.ipc021", False),   # a denial we did not ask to lift
    ("xiaomi.wifispeaker.oh11", True),   # already allowlisted
    ("chuangmi.camera.066a01", True),    # never denied
]


async def main():
    failures = []

    baseline = await load("")
    for model, want, _ in CASES:
        got = passes(baseline, model)
        if got != want:
            failures.append("baseline %s: expected %s, got %s" % (model, want, got))
    for model, want in UNTOUCHED:
        got = passes(baseline, model)
        if got != want:
            failures.append("baseline %s: expected %s, got %s" % (model, want, got))

    overridden = await load(",".join(model for model, _, _ in CASES))
    for model, _, want in CASES:
        got = passes(overridden, model)
        if got != want:
            failures.append("overridden %s: expected %s, got %s" % (model, want, got))
    for model, want in UNTOUCHED:
        got = passes(overridden, model)
        if got != want:
            failures.append("overridden %s: expected %s, got %s" % (model, want, got))

    if failures:
        for line in failures:
            print("FAIL", line)
        raise SystemExit(1)
    print("CAMERA_ALLOW_MODELS behaves as expected for %d models" % (len(CASES) + len(UNTOUCHED)))


asyncio.run(main())
