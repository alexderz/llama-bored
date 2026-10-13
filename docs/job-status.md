# Job status files

A batch job (a training run, an evaluation) writes one JSON document into a
directory. llama-watch reads those files and, while a job is live and no
model is serving, shows a job view in place of the inference panels.

Nothing about a job reaches the screen that the job did not put in its
file. llama-watch does not parse a log, inspect a process, or open any
other path.

## Config

```toml
[jobs]
dir = "/var/lib/llama-bored/jobs"   # absolute; unset opens nothing
stale_s = 180                       # 10..=3600; a file older than this is not shown
```

Leave `dir` unset and llama-watch opens no job file. Every console frame
and the grid file are then byte-identical to a build without this feature.

## File

One file per job, named `<name>.json`, replaced atomically (write a
temporary file in the same directory, then rename). Schema 1:

```json
{
  "schema": 1,
  "name": "example-trainer",
  "kind": "training",
  "title": "Example-2B",
  "state": "running",
  "phase": {"label": "2B", "index": 2, "of": 3},
  "step": 921, "total": 1075,
  "elapsed_s": 15381, "eta_s": 2559,
  "rate": {"value": 16.62, "unit": "s/step"},
  "rates": [{"label": "rows/s", "value": 0.96}, {"label": "tok/s", "value": 1540.0}],
  "values": [{"label": "loss", "value": 0.2544}, {"label": "lr", "value": 1.276e-05},
             {"label": "epoch", "value": 0.8465}, {"label": "grad", "value": 0.348}],
  "series": [{"label": "loss", "points": [3.285, 0.846, 0.539, 0.254]},
             {"label": "s/step", "points": [16.4, 16.7, 16.6]}],
  "phases": [{"label": "synthetic test", "state": "done", "elapsed_s": 356, "result": "pass"},
             {"label": "0.8B", "state": "done", "elapsed_s": 10186, "result": "val 98.2 %"},
             {"label": "2B", "state": "running", "elapsed_s": 15381},
             {"label": "4B", "state": "queued"}]
}
```

Required: `schema` (must be 1), `name`, `state` (`running`, `done` or
`failed`). Everything else is optional; the view shows what is there.
Unknown members are ignored. `kind` is a short token
(`[A-Za-z0-9_.+-]`, at most 16 characters).

Caps: `rates` 4, `values` 8, `series` 4 with at most 512 points each
(newest kept), `phases` 12. Text is sanitised like SETUP `text` (controls
and escape sequences dropped) and cut: name 32, title 48, labels 16, units
12, result 32 characters. A non-finite or out-of-range number makes that
one member absent, not the whole file invalid. `step`, `total`,
`elapsed_s` and `eta_s` are non-negative.

llama-watch looks at at most eight `*.json` files directly in `dir`, opens
them without following a symlink, rejects a body over 64 KiB or JSON
deeper than 8, and re-reads at most once a second and only when the mtime
or size changed. A file that fails a check is skipped and logged once.

A job is live when its file is younger than `stale_s` and `state` is
`running`. With several live jobs the view shows the one whose file
changed most recently, with `+N` after the name. When a model is serving
(llama-swap's `/running` list is not empty), the inference view stays and
a one-line note names the live job. When the last live job is `done` or
`failed` and still fresh, that note is `name done 4 h 16 min`.

## Reference writer

A Hugging Face `transformers` Trainer callback. It is documentation, not
installed with llama-bored.

```python
import json
import os
import tempfile
import time


class JobStatusCallback:
    """Write schema 1 status for llama-watch. Pass as a Trainer callback."""

    def __init__(self, dir, name, title, kind="training", total=None):
        self.path = os.path.join(dir, f"{name}.json")
        self.name = name
        self.title = title
        self.kind = kind
        self.total = total
        self.t0 = time.monotonic()
        self.loss = []
        self.step_t = []
        self.last = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        self._write(state, logs)

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        self.step_t.append(now - self.last)
        self.last = now
        if len(self.step_t) > 512:
            self.step_t = self.step_t[-512:]

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state, {}, done=True)

    def _write(self, state, logs, done=False):
        elapsed = max(0, int(time.monotonic() - self.t0))
        step = int(getattr(state, "global_step", 0) or 0)
        total = self.total or getattr(state, "max_steps", None)
        loss = logs.get("loss")
        if isinstance(loss, (int, float)):
            self.loss.append(float(loss))
            if len(self.loss) > 512:
                self.loss = self.loss[-512:]
        rate = self.step_t[-1] if self.step_t else None
        eta = None
        if rate and total and step < total:
            eta = int(rate * (total - step))
        body = {
            "schema": 1,
            "name": self.name,
            "kind": self.kind,
            "title": self.title,
            "state": "done" if done else "running",
            "step": step,
            "elapsed_s": elapsed,
        }
        if total:
            body["total"] = int(total)
        if eta is not None:
            body["eta_s"] = eta
        if rate is not None:
            body["rate"] = {"value": rate, "unit": "s/step"}
        values = []
        for key in ("loss", "learning_rate", "epoch", "grad_norm"):
            val = logs.get(key)
            if isinstance(val, (int, float)):
                label = {"learning_rate": "lr", "grad_norm": "grad"}.get(key, key)
                values.append({"label": label, "value": float(val)})
        if values:
            body["values"] = values
        series = []
        if self.loss:
            series.append({"label": "loss", "points": list(self.loss)})
        if self.step_t:
            series.append({"label": "s/step", "points": list(self.step_t)})
        if series:
            body["series"] = series
        raw = json.dumps(body, ensure_ascii=True)
        directory = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(prefix=".job-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
```

Use it as `Trainer(..., callbacks=[JobStatusCallback("/var/lib/llama-bored/jobs", "example-trainer", "Example-2B", total=1075)])`.
