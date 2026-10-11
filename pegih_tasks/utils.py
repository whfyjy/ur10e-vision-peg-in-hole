"""Small helpers shared by the scripts. Standard library only, so it is safe to
import before AppLauncher starts Isaac Sim."""

from __future__ import annotations

import threading


def safe_close(obj, timeout: float = 15.0, label: str = "close()") -> None:
    """Call obj.close() on a daemon thread and give up after `timeout` seconds.

    Isaac Sim's teardown can hang indefinitely. Putting an os._exit() fallback in
    __main__ is not enough on its own, because the usual shape is

        def main():
            ...
            env.close()        # <- hangs HERE, inside main()

        if __name__ == "__main__":
            main()
            simulation_app.close()   # <- never reached

    A camera_snapshot.py run sat in env.close() for 30 hours holding 3.6 GB of
    VRAM that way, and the next training run died with CUDA OOM because of it.
    Wrapping the close itself is what actually bounds the wait.

    Losing whatever the close would have flushed is acceptable: the process is
    about to exit anyway, and every artefact these scripts produce (images,
    checkpoints, logs) is already written by then.
    """
    t = threading.Thread(target=obj.close, daemon=True, name=f"safe_close:{label}")
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"  [{label}] did not return within {timeout:.0f}s; continuing anyway", flush=True)
