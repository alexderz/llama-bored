# llama-bored

**It's for watching when you're bored waiting on your agent.**

Live telemetry from a local-AI box on Linux: llama.cpp through
[llama-swap](https://github.com/mostlygeek/llama-swap), GPU, CPU, power and
temperatures. It shows up on a full-screen tty11 dashboard, on the 320x320
round LCD of an NZXT Kraken Z cooler, on your case RGB, and on a Prometheus
endpoint. Your agent runs a 20-minute job on a local model; you get something
to look at while it does.

It never touches the pump or fans. Everything it draws on hardware is colour
and pixels, through small closed command sets.

It needs Linux with systemd. Everything else is optional: an NZXT Kraken Z
(`1e71:3008`) for the LCD, NVIDIA, llama-swap, ASUS Aura and the Corsair
keyboard. Without a Kraken Z the installer sets up everything except the
LCD writer.

<p align="center">
  <img src="docs/media/kraken-lcd.gif" width="560" alt="The Kraken LCD while a new model loads and starts generating: the activity gauge swings from idle into the redline">
</p>

<p align="center">
  <img src="docs/media/tty11.gif" alt="The tty11 dashboard during the same 12 seconds">
</p>

<sub>Both recorded from the same 12 seconds on a real box: one model unloads,
Qwen3-Coder loads onto the GPU, and generation ramps to about 160 tok/s. The
LCD frames are rendered by kraken-lcd from recorded snapshots and set into a
drawn pump-head scene; the tty11 frames come from the console's own screen
buffer.</sub>

## Components

| Binary | Unit | What it does |
|---|---|---|
| `llama-watch` | `llama-watch.service` | The **single reader** of host and llama state. Draws the tty11 dashboard (Ctrl+Alt+F11 from a graphical session, Alt+F11 from another console) and publishes a validated snapshot to `/run/llama-watch/snapshot.json` at 10 Hz |
| `llama-view` | none | Read-only mirror of tty11 for tmux or SSH (group `llama-view`) |
| `kraken-lcd` | `kraken-lcd.service` | Draws the snapshot on the NZXT Kraken Z LCD. LCD only, no network |
| `llama-light` | `llama-light.service` | Sets ASUS Aura and Corsair keyboard RGB colours from the snapshot. Colour only, no network |
| `llama-metrics` | `llama-metrics.service` | Prometheus exporter on port 19477 (`llamabored_*`). Off by default |
| `llama-cast` | `llama-cast.service` | tty11 as a live video for TVs on your LAN (DLNA). Off by default |

`llama-core` is the shared library: the snapshot schema and its validator, the
name sanitiser, logging.

**Engine.** tty11's header always names the inference engine serving the
shown model: `llama.cpp`, `SGLang`, `vLLM`, `Strata`, `OpenAI-compatible`
(a llama.cpp fork is `llama.cpp`); with no model loaded it says
`engine --`. Its settings are in the SETUP block (below). The LCD starts the
line under the model name with the short name (`llama.cpp`, `sglang`,
`vllm`, `strata`, `openai`) and drops the quant first when space is short.
The exporter carries the wire word (`llamacpp`, `sglang`, `vllm`, `strata`,
`openai`) as the `engine` label on every per-model series (see
**llama-metrics** under Features).

**Speeds.** RECENT's PROMPT and GEN tok/s come from llama-swap, which has
them only for llama.cpp. For a vLLM request they are measured by the
engine instead and marked `~` (`~2,134`, `~41.3`): llama-watch reads vLLM's
per-request prefill and decode time histograms on every metrics poll and
gives a new row the speeds of the requests that finished around it —
uncached prompt tokens over prefill time, and generated tokens after the
first over decode time (speculative decoding included). If several
requests finished in that window, the row shows their average; if none
did, it keeps `--`. Nothing is derived from DURATION. SGLang reports no
per-request prefill or decode time, so its rows keep `--`. The same window
speeds are at the end of the model's backend line on tty11 when it fits;
the exporter carries the counters they come from instead
(`llamabored_model_prefill_seconds_total` and friends, see
**llama-metrics** under Features). Strata reports no histograms
but its lifetime `totals` (#54): prefill = Δ(prompt − reused tokens) ÷
Δ`prompt_ms`, decode = Δoutput tokens ÷ Δ`decode_ms` (every output token,
as Strata times them), over the same windows; a total going down is a
restart and its new totals are the window. Its RECENT rows keep
llama-swap's own speeds.

## Features

**llama-watch and tty11** (10 fps, bundled Hack console fonts: 12x24, 12x22, 10x18)

- Spectrum meters for CPU, GPU, VRAM, MEM, POWER and LOAD, and an **ACTIVITY** row that
  names its source (`gpu`, `cpu` or `util`).
- Prompt and generation tok/s, and a token chart in eighth-block bars
  (generation up, prompt down, newest on the left), log-scaled to
  `gen_ceiling_tps` and `prompt_ceiling_tps`.
- **SLOTS:** per-slot context fill and a context-history sparkline (6 h by
  default) with a marker where a context dropped, labelled by best-guess
  reason from token counts: `c` compacted (same conversation, mostly
  cached), `n` new conversation, `e` evicted (a conversation came back with
  nothing cached), red `v` unknown.
- **RECENT:** the last requests across the full width, with timestamps, a
  context bar per request, and in-flight rows while a request runs (below).
- **IN / OUT:** the live prompt and output tails of the busy slot. IN strips
  chat-template tokens by default. For SGLang, vLLM and other servers
  without `/slots`, IN/OUT show the last finished exchange from llama-swap's
  request captures (when its `captureBuffer` is on; read up to 2 MiB per
  capture). Turn all text off, captures included, with `show_text = false`.
- **TEMPS:** every temperature the host exposes, one row per device, with
  allow and block lists (see **TEMPS and FANS** under Configuration).
- **FANS** (optional): read-only rpm and pwm of discovered fans, with the
  same allow and block lists.
- The model's full name and engine in the header, and a **SETUP** block with
  the loaded model's settings (below).

**Context bar.** RECENT's bar (#75) shows how much of the model's
context each request used, as three runs in a row: cached input (blue),
new input (input minus cached; sky blue) and output (magenta). The full
width is the model's context size (`n_ctx` from the engine or the launch
command, remembered after the model unloads), so a cell is `n_ctx / width`
tokens, linearly. Each run starts with its remainder cell: its height is
the fraction of a cell in eighths (`▁`..`▇`, at least `▁` for any tokens at
all), and its colour the fraction of an eighth, in this fixed order of
eight palette slots, the same on tty11, in llama-view (16, 256 and
truecolor) and on llama-cast: red, orange, amber, yellow, light green,
green, sky, blue (slots 1, 6, 3, 11, 10, 2, 14, 4): red is the smallest step,
blue the largest. At 30
cells and a 262,144-token context one colour step is about 136 tokens.
The RECENT header row carries the order as a key (#77): the eight
colours as full blocks, left-aligned directly over the bar's first cells,
so a remainder cell's colour reads as the start, middle or end of its
eighth by matching it to the key; a bar narrower than eight cells shows
what fits. The legend keeps `█cached █new █out`.
The colours are a function of the counts only: a finished row never
changes. Empty cells are a dim baseline. A request at 90 % of the context
or more gets a yellow `!` after its bar. Without a context size the bar is
scaled to the largest such row shown and marked `~`. With `chart_glyphs =
"halves"` the heights fall back to `▄` and `█` and the colours carry the
rest. Engines that report no cached count show all input as new.

```text
 RECENT TIME      SOURCE      MODEL          IN   CACHED   OUT  ...    DUR ████████
   4822 18:47:01  192.0.2.83  Qwen 35B   91,204   88,960   612  ...  13.2s ▁███████████▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
>       18:47:20              Qwen 35B  120,000        0     0  ...  36.4s ▁██████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ pp c
```

**In-flight rows.** Where the engine reports a request while it runs, it
is RECENT's top row until llama-swap's finished row replaces it (never
both): llama.cpp from `/slots` (one row per busy slot, at most half the
rows), Strata from its `live` report. IN is the whole prompt, CACHED the
reused part, OUT the tokens so far, DUR the time since it was first seen;
PROMPT and GEN are the live rates, marked `~`. During prefill the new run
fills, poll by poll, toward the rest of the prompt drawn as a sky-blue low
line. llama.cpp's `/slots` does not give the prompt's length, so its
prefill row shows only what the slot holds so far, IN with a `+`
(`69,632+`) and no low line; it turns exact when decoding starts. The
prefill row is marked `pp`, with the SLOTS reset letter (`c`, `n`,
`e`) when a context reset just made it start from zero; then `gen` while
it decodes. The row holds the latest poll's numbers, nothing in between.
vLLM, SGLang and OpenAI-compatible servers report no per-request
progress: their newest finished row is marked `gen` while the model
generates, as before.

**SETUP.** Under the meters (one per row since #52), tty11 shows the
settings of the model generating now, or else the one RECENT saw last:

```text
  CPU           41 %  16c  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  GPU                97 %  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░
  VRAM       22.8/24.0 GB  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░
  MEM        38.1/62.6 GB  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░░░░░░
  POWER         312/350 W  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░░░░
  LOAD               70 %  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░
  ACTIVITY 42 % gpu 312 W  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

  SETUP  qwen3.6-35b-a3b · Qwen 35B                                         +1
    engine   llama.cpp · UD-Q4_K_M · fa on
    ctx      262,144 · kv q8_0 / q8_0
    experts  16 layers in RAM
    spec     draft-mtp · n-max 3
    think    budget 24,000
```

The title is the llama-swap model id, then its `name` when that fits, and
`+N` when N more models are loaded. Each row is a setting read from the
model's llama-swap launch command, or from what the engine reports (KV
dtype, block size and prefix caching from vLLM's `cache_config_info`,
Strata's context, KV and the rest of its `engine` settings, live
speculative acceptance `acc 3.6/step · 65 %`). A Strata model reads:

```text
    engine   Strata 0.1.41 · Q4_K_M
    ctx      262,144 · kv q8 · resident 24,576
    experts  cache 14.6 GiB · 7,200 slots · hit 87 % · pcie 9 %
    spec     depth 5 · mtp 3 · lookup 2 · min-p 0.40 · 70 %
    serve    pcie 0.60 · arena 40.5 GiB · 12 workers · conv cache off
```
A value nothing set is grey (`kv f16`, `full GPU`, `spec none`). Which
settings show is configuration: see **SETUP rules** under Configuration.
The block has six rows (title included) up to 59 rows and eight from 60;
rows that do not fit drop from the bottom, and items that do not fit drop
from the end of their row. Under 48 rows it uses only the rows beside
SLOTS, so the short-screen panel order is unchanged. With nothing loaded,
or llama-swap down, it is gone.

The meter bars are whole cells of `▇` (lower seven eighths) when
`tty.chart_glyphs = "eighths"`, which needs the llama-hack font, so
stacked bars keep a gap; with `"halves"` (a console font such as eurlatgr,
which has no `▇`) they use `▄`. 4K screens (90+ rows) keep their two-row
bars.

**Activity** is the bottleneck device's power headroom. For the GPU (NVML) and
the CPU (zenergy socket energy), `(watts − idle) / (nominal_frac × limit −
idle)`; the larger wins. 100 is sustained heavy load, spikes reach 125. Idle
floors are learned from 10 s means and never rise. Without power sensors it
falls back to utilisation.

**kraken-lcd** ("V3b Plasma Blackbody" face)

- A 270° activity gauge, 0–125, with a 100–125 redline where spikes glow
  blackbody-hot and a pegged gauge throws smoke and embers.
- An inner dial of 24 bars with 30 minutes of activity history and a time
  scale; a tokens·24h chart; model name and tuning line; coolant, CPU and GPU
  temperatures; CPU and memory.
- **LCD tokens chart.** Under the temperatures the LCD (layout A1) plots
  decoded tok/s over 24 hours, newest on the left, scaled to a 1·2·5 ceiling
  printed top left. The current generation rate floats large over the
  chart's far end, with `tok/s` above it (#55): a 5 s moving average of the
  decoded-token counter, `7.5` under 10, `112` up to 999, then `1.2k` and
  `12k`. It reads `0` (grey) once no token has been decoded for 3 s, as
  during prompt processing or when idle, and `—` with no counter or no data. In `upload.mode = "change"` it
  moves in its printed steps with a 30 % margin, so the panel re-uploads
  only when the text changes; stream mode shows it every frame.
- Distinct "AI down", "no model" and "no data" states. After 30 s without data
  the LCD returns to the stock coolant screen.
- *Change* mode (default: upload only when a drawn value changes, at most once
  per `min_interval_s`) or opt-in *stream* mode (10 fps, animated gauge).

**llama-light** (ASUS Aura USB RGB, Corsair STRAFE RGB MK.2 keyboard)

- Maps any snapshot metric (activity, gpu, cpu, cpu_topk (the busiest
  cores), load, mem, tokens_rate, coolant, gpu_temp, cpu_temp) to colour,
  with `[[light]]` layers: solid, ring, bar or pulse styles, palettes or your own colour stops, smoothing,
  brightness caps and an idle colour.
- Fans on a splitter (mirrored) or a daisy chain (per-fan values).
- Per-key keyboard gauges: bars two key rows tall, a tokens/s dial on the
  number pad, with an optional tweened frame pipeline so values glide
  instead of flicker. The example file has a full layout. On stop the
  keyboard gets its own lighting back.

  <img src="docs/media/keyboard.gif" alt="The keyboard layout during the same 12 seconds: activity on the F-row, GPU and busiest cores as two-row bars, GPU temperature on the navigation keys, tokens/s as a dial on the number pad">

  <sub>The keyboard layout during the same 12 seconds, computed by llama-light
  from the recorded snapshots and drawn with simulated keycap bleed.</sub>
- Live reload: edits to `light.toml` apply within 2 s; a bad file is logged
  and ignored.

**llama-metrics** (Prometheus)

- `GET /metrics` on `:19477` for a CIDR allowlist. Numbers and model names
  only; never prompt or output text. Every snapshot field is exported (a
  test fails if one is not).
- It is the one scrape target: never scrape the engines through
  llama-swap's `/upstream/<id>/…`, which can load a model (#70).
- Since 0.5 (#71) every engine's numbers that mean the same thing are one
  series, told apart by labels: `model` (the llama-swap model id, exact;
  the join key), `engine` (`llamacpp`, `vllm`, `sglang`, `strata` or
  `openai`), and `slot`, `state`, `status` and `reason` where they apply.
- Names carry their unit, ratios are 0 to 1, only counters end in
  `_total`, and descriptive strings live on `llamabored_model_info` only.
  Counters count since llama-watch started: an engine restart or a model
  unloading and coming back carries on from the total, and only a
  llama-watch restart resets them (which `rate()` handles). A series an
  engine cannot report is absent, not zero; a model that is not loaded has
  no series at all, while the host series stay.

Per model (labels `model`, `engine`):

| Series | Type | Filled by |
|---|---|---|
| `llamabored_model_info{display_name, quant, kv_type, version}` | gauge, 1 | every engine (`version`: Strata; empty when unknown) |
| `llamabored_model_state{state}` | gauge | every engine (`ready`, `starting`, `stopping`, `other`) |
| `llamabored_model_context_size_tokens` | gauge | `-c`, `--max-model-len`, `--context-length`, Strata `context` |
| `llamabored_model_requests_running`, `…_requests_waiting` | gauge | llama.cpp (`requests_processing`, `requests_deferred`), vLLM, SGLang, Strata |
| `llamabored_model_slots` | gauge | llama.cpp slots; vLLM `--max-num-seqs`, SGLang `--max-running-requests`, Strata 1 |
| `llamabored_model_kv_cache_usage_ratio` | gauge | vLLM, SGLang; llama.cpp builds that still report it |
| `llamabored_model_prompt_tokens_total` | counter | every engine (llama.cpp and OpenAI-compatible from llama-swap's activity rows) |
| `llamabored_model_prompt_cached_tokens_total` | counter | llama.cpp (`cache_tokens`), vLLM, SGLang, Strata (`reused`) |
| `llamabored_model_generation_tokens_total` | counter | every engine (llama.cpp `tokens_predicted_total`; OpenAI-compatible from activity rows) |
| `llamabored_model_prefill_seconds_total`, `…_decode_seconds_total` | counter | llama.cpp (`prompt_seconds_total`, `tokens_predicted_seconds_total`), vLLM (per-request time sums), Strata (`prompt_ms`, `decode_ms`) |
| `llamabored_model_requests_total{status}` | counter | every engine, from llama-swap's activity rows (`ok` is 2xx, else `error`) |
| `llamabored_model_time_to_first_token_seconds` | summary (`_sum`, `_count`) | vLLM, SGLang |
| `llamabored_model_inter_token_latency_seconds` | summary | vLLM, SGLang |
| `llamabored_model_request_duration_seconds` | summary | vLLM, SGLang (e2e histograms); others from activity row durations |
| `llamabored_model_spec_draft_tokens_total`, `…_spec_accepted_tokens_total` | counter | vLLM, Strata, llama.cpp (activity draft fields) |
| `llamabored_model_spec_drafts_total` | counter | vLLM (the others count no draft rounds) |
| `llamabored_model_preemptions_total`, `llamabored_model_sleeping` | counter, gauge | vLLM |
| `llamabored_model_kv_block_size_tokens`, `llamabored_model_prefix_caching` | gauge | vLLM `cache_config_info` |
| `llamabored_model_expert_cache_hit_ratio`, `…_pcie_share_ratio` | gauge | Strata: the newest finished request's `hit_rate`, `pcie_share` |
| `llamabored_slot_context_used_tokens` | gauge | llama.cpp; labels `model`, `engine`, `slot` |
| `llamabored_slot_context_resets_total{reason}` | counter | llama.cpp; labels `model`, `engine`, `slot`; best-guess reason (compacted, new, evicted, unknown) |

The latencies are `summary` families with no quantiles: one `# TYPE …
summary` and a `_sum` and `_count` sample per model, both counters to
`rate()`. Not reported, so absent: TTFT and ITL for llama.cpp, Strata and
OpenAI-compatible servers; prefill and decode seconds for SGLang and
OpenAI-compatible servers; speculative counters for SGLang (it reports
only gauges) and OpenAI-compatible servers.

Host and exporter:

| Series | Labels | Meaning |
|---|---|---|
| `llamabored_activity_ratio` | | Activity, 0 to 1.25 (1 is nominal sustained load) |
| `llamabored_load_ratio`, `_cpu_utilization_ratio`, `_cpu_topk_utilization_ratio`, `_gpu_utilization_ratio` | | Host load, 0 to 1 |
| `llamabored_coolant_celsius`, `_cpu_celsius`, `_gpu_celsius` | | Temperatures |
| `llamabored_gpu_power_watts`, `_gpu_power_limit_watts`, `_cpu_power_watts` | | Power draw and GPU limit |
| `llamabored_gpu_memory_used_bytes`, `_total_bytes`; `llamabored_memory_used_bytes`, `_total_bytes` | | VRAM and system memory |
| `llamabored_ai_state` | `state` | down / idle / loaded |
| `llamabored_temperature_celsius` | `chip`, `sensor` | Every temperature TEMPS shows, the GPU as `chip="gpu"` (#74); the coolant, CPU and GPU series stay |
| `llamabored_fan_rpm`, `llamabored_fan_pwm_ratio` | `chip`, `channel`, `label` | FANS panel, when `[fans]` is on (read only) |
| `llamabored_source_up`, `llamabored_source_latency_seconds` | `source` | Health of each watcher source (llama-swap, running, slots, metrics, activity, gpu, hwmon, proc) |
| `llamabored_collector_suspected_loads_total` | `model` | Suspected model loads after an upstream read (#70); absent while there are none |
| `llamabored_snapshot_up`, `_snapshot_stale`, `_snapshot_age_seconds`, `_snapshot_seq` | | Snapshot freshness |
| `llamabored_exporter_build_info`, `_exporter_rejected_connections_total` | | Exporter version and refusals |

Examples:

```promql
# Decode tok/s while generating, per model
rate(llamabored_model_generation_tokens_total[5m])
  / rate(llamabored_model_decode_seconds_total[5m])

# Prompt cache hit ratio
rate(llamabored_model_prompt_cached_tokens_total[5m])
  / rate(llamabored_model_prompt_tokens_total[5m])

# Speculative acceptance
rate(llamabored_model_spec_accepted_tokens_total[5m])
  / rate(llamabored_model_spec_draft_tokens_total[5m])

# Mean time to first token, seconds
rate(llamabored_model_time_to_first_token_seconds_sum[5m])
  / rate(llamabored_model_time_to_first_token_seconds_count[5m])
```

Prefill tok/s is `rate(prompt_tokens_total - prompt_cached_tokens_total)`
over `rate(prefill_seconds_total)` (every engine times only the tokens it
computed). Add a display name with `* on (model, engine)
group_left(display_name) llamabored_model_info`.

**Dashboards need a migration for 0.5**: every per-model series changed
its labels (`model` and `engine` instead of `name` and `full_name`) and
many changed names. [CHANGELOG.md](CHANGELOG.md) has the old-to-new table.

**Backends behind llama-swap.** llama-bored tells each model's server apart
by its launch command (or `[llama.backends]` in `watch.toml`):

| Backend | What you get |
|---|---|
| llama.cpp `llama-server` and forks (ik_llama.cpp, PrismML) | Everything: tok/s, SLOTS with context fill and reset reasons, live IN/OUT text (needs `LLAMA_SERVER_SLOTS_DEBUG=1` on current llama-server; without it IN/OUT show the last finished exchange from llama-swap captures, #66), the tuning line |
| SGLang | tok/s, running and queued requests, KV fill and cache hit rate from `sglang:*` metrics (start it with `--enable-metrics`); tuning line from its flags; IN/OUT from llama-swap captures |
| vLLM | The same from `vllm:*` metrics, plus speculative-decoding acceptance (`spec 78 %`, also on the LCD), mean TTFT, inter-token and request latency, preemptions, and the KV dtype, block size and prefix caching from `cache_config_info`; IN/OUT from llama-swap captures |
| Strata (`serve/server.py --engine strata`, or recognised by its JSON `/metrics`) | From its JSON `/metrics`: tok/s, running and queued requests (one at a time), its settings in SETUP (context, KV, expert cache, spec depth, serving knobs), speculative acceptance, window prefill and decode tok/s, expert cache hit rate and PCIe share, and its live phase (`writing a tool call: write`, tty11 only, never exported); numbers and short tokens only, at most 1 MiB, `history` never read; no KV fill or cache hit rate; IN/OUT from llama-swap captures. Started with `--api-key`, it falls back to llama-swap's request log (llama-bored keeps no keys) |
| Any other OpenAI-compatible server (TabbyAPI, ...) | Token counts from llama-swap's request log; GPU, CPU and activity as always |

When the launch command does not name the server (a container whose image
starts it, or a wrapper script), llama-watch reads that model's `/metrics`
once while it is loaded and recognises the server by its metric names
(`vllm:`, `sglang:`, `llamacpp:`), or Strata by the shape of its JSON (an
object whose `engine` and `live` members are objects). It only ever reads models llama-swap
reports as `ready`, so it never makes llama-swap load one (see
**Never swaps models** below).
`[llama.backends]` overrides the detection. Once the server is known either
way, its flags are read from the arguments after the image of a
`podman run` / `docker run` command (the whole command when no image is
found), so a llama.cpp image started by digest still gets its ctx, KV,
`-ncmoe` and quant in SETUP and on the exporter (#67).

IN and OUT show a llama.cpp model's live prompt and output from `/slots`.
Current llama-server sends that text only when it runs with
`LLAMA_SERVER_SLOTS_DEBUG=1`; an env var set on llama-swap does not reach a
server in its own container, so pass it there with `-e`. Without it IN and
OUT show the last finished exchange from llama-swap's request captures and
the log says so once per model; SLOTS, ctx and resets still come from
`/slots`, and a server restarted with the variable gets live text again.

**Never swaps models (#70).** llama-swap starts, or swaps in, any model
an `/upstream/<model>/…` request names that is not loaded, so llama-watch
makes sure its own polling never does:

- **Fresh gate.** Every upstream read (`/metrics`, `/slots`, the one-time
  engine probe) is sent right after a `/running` read of its own, and only
  for a model that read lists as `ready`. A model that was ready a moment
  ago but is not in that read is skipped, not retried.
- **No reads during a swap.** If that `/running` lists any model as
  `starting`, `stopping` or in any other state, no upstream request is
  made for the rest of that round.
- **Self-check.** `/running` is read again just after each upstream read.
  A model `starting` there, or an upstream read slower than 2 s, is a
  suspected load: a warning names the model and path
  (`<id>: suspected model load after GET /upstream/<id>/metrics (…)`), that
  model gets no upstream reads for 5 minutes, and
  `llamabored_collector_suspected_loads_total{model}` counts it.
- A `409`, or any other non-2xx answer, from an upstream read means "not
  available now": that model's engine series are dropped as on unload,
  it is not read again that round, and llama-swap's `source_up` stays 1.
  A 409 is logged once (`<id>: llama-swap says not loaded; skipping until
  ready`). Redirects are never followed.

When a model is not running, its series are simply absent: llama-metrics
renders only what the snapshot holds, so the exposition stays valid and
`llamabored_ai_state` is still there.

llama-swap still decides between our `/running` read and our upstream
read, so a short window remains (one request apart, normally well under
a millisecond) in which another client's request can start a swap. As optional extra protection, list the
poller's paths in llama-swap's `upstream.ignorePaths` (v256 and later):
llama-swap then answers such a request for a model that is not loaded
with `409` and never loads it. Listing `ignorePaths` replaces llama-swap's
default pattern, so keep that one too:

```yaml
upstream:
  ignorePaths:
    - '.*\.(js|json|css|png|gif|jpg|jpeg|ico|txt)$'
    - '^/(metrics|slots|props|health)$'
```

**Works without AI:** llama-swap, NVIDIA, the power sensors and the Aura
controller are optional; missing sources show "—".

## Supported hardware

| Device | USB id | Status |
|---|---|---|
| NZXT Kraken **Z53** | `1e71:3008` | **Tested** (kraken-lcd) |
| NZXT Kraken Z63 / Z73 | `1e71:3008` (same id) | Same LCD protocol, accepted, **untested** |
| Other NZXT (Kraken 2023/Elite, X-series) | other ids | Not supported; kraken-lcd refuses them |
| ASUS Aura USB mainboard controller | `0b05:18f3` | **Tested** (llama-light, addressable header 1) |
| Corsair STRAFE RGB MK.2 | `1b1c:1b48` | **Tested** (llama-light, per key, lighting interface only) |
| Other Corsair keyboards | other ids | Not supported |

The Z63 and Z73 report the same USB id and `z53` hwmon name as the Z53, so no
software check can tell them apart. If you try one, read
[docs/SAFETY.md](docs/SAFETY.md) first and stay close for the first run.

## Requirements

- Linux with **systemd**.
- For the LCD (optional): one NZXT Kraken Z and the **`nzxt-kraken3`** hwmon
  driver (mainline since 6.9) bound to it. With none attached the installer
  skips kraken-lcd's unit, udev rules and state directory (the binary is
  still installed) and says how to add it later: attach the cooler and run
  the installer again. With more than one it refuses.
- **Rust** via rustup (`rust-toolchain.toml` pins the version), plus
  `cargo-deny` and `cargo-audit` for `scripts/check.sh`.
- For the build and install scripts: `git`, a C compiler as the linker
  (`cc`), `nm` (binutils), `ldd` (glibc), `sha256sum` (coreutils),
  `setfacl`/`getfacl` (acl), and udev and `systemd-sysusers`. `python3` is
  optional (the font rebuild check is skipped without it).
- Optional: an **NVIDIA** GPU and driver (NVML); **llama-swap** on loopback
  (`llama-server --metrics` for token rates, `LLAMA_SERVER_SLOTS_DEBUG=1` for
  live text); **zenergy** (AMD socket power); **k10temp** or **coretemp**; a
  **Super-I/O hwmon** (`nct67xx`, `it87`, ...) for FANS.

## Safety model

The Kraken's USB device drives the LCD **and** the CPU pump and fans; the Aura
controller also runs fan-header lighting. The design treats cooling as the
main risk:

- **Closed command sets.** kraken-lcd can build nine LCD commands; pump, fan,
  init, brightness and firmware commands do not exist in the code.
  llama-light can build two Aura opcodes (set direct mode, stream colours) and
  four keyboard report shapes, with no save-to-flash, profile or firmware
  writes. Tests check all three tables.
- **Cooling guard and HALTED latch.** Every LCD operation is bracketed by
  read-only checks of the cooler's hwmon (pump mode, pump rpm, USB device,
  bootloader). Any change stops all device I/O and writes a latch file that
  stays until an explicit root command clears it.
- **Nothing writes sysfs.** The FANS panel only reads.
- **Split privileges.** One reader with no device access; writers with no
  network and only their own device nodes. kraken-lcd's hidraw grant is
  pinned by udev and `DeviceAllow=` to the cooler's node; its usbfs grant is
  class-wide in the unit, and udev gives only the cooler's usbfs node to its
  group. llama-light is pinned the same way to two hidraw nodes, the Aura
  controller and the keyboard's lighting interface. One exporter that opens
  no device. Only llama-metrics and llama-cast listen, both off by default.
- **Stock restore on stop.** Stopping or crashing kraken-lcd gives the screen
  back to the stock readout; llama-light leaves a neutral colour in RAM and
  hands the keyboard back to its own lighting.

Details: [docs/SAFETY.md](docs/SAFETY.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install (the lazy way)

Point your AI agent (with root, and you nearby) at
[`skills/install/SKILL.md`](skills/install/SKILL.md): *Install llama-bored on
this machine following skills/install/SKILL.md*. It checks the machine, builds
and verifies as your normal user, writes the configs from what it detects
(asking you about text panels, fans, RGB, the exporter and stream mode),
installs, and does a guarded first LCD write. It has hard rules against
touching cooling.

## Manual install

As your normal user; `sudo` only where shown.

```sh
git clone https://github.com/alexderz/llama-bored && cd llama-bored
git switch main
scripts/check.sh                  # fmt, clippy, tests, deny, audit, safety fences, release build
scripts/cooling-snapshot.sh | tee ~/cooling-before.txt
sudo scripts/install.sh           # add --enable-light to enable RGB for boot; save the rollback text
```

`install.sh` installs only a build that `check.sh` produced from a **clean
tree whose HEAD is on `main`**, with matching binary hashes. It creates the
users and udev rules, creates `/var/lib/kraken-lcd`, enables `llama-watch`,
and **never starts kraken-lcd**. llama-light is enabled only with
`--enable-light`; llama-metrics is never enabled. A file it replaces with different content is kept as
`<file>.bak-<previous sha>`; an unchanged file gets no backup, and it keeps
only the newest three of its own backups per file. Backups you name yourself
(`watch.toml.bak-pre-setup`) are left alone. Then, as root:

```sh
systemctl start llama-watch                           # tty11: Ctrl+Alt+F11
d=$(mktemp -d) && chmod 0755 "$d" && install -m 0644 fixtures/views/test-card.json "$d/"
runuser -u kraken-lcd -- /usr/local/libexec/llama-bored/kraken-lcd show-image --view "$d/test-card.json"
```

The LCD shows a "TEST IMAGE" card. Run `scripts/cooling-snapshot.sh` and
compare: `pwm1_enable`/`pwm2_enable` unchanged, pump rpm about the same. Then:

```sh
sudo systemctl enable --now kraken-lcd
sudo systemctl start llama-light                         # optional RGB; enable to keep it
sudo systemctl --global enable kraken-lcd-halt.path      # optional desktop alert on HALTED
sudo usermod -aG llama-view "$USER"                      # optional, for llama-view; log in again
```

Installed: binaries in `/usr/local/libexec/llama-bored/` and
`/usr/local/bin/llama-view`, the font in `/usr/local/share/llama-bored/`,
configs in `/etc/llama-bored/`, units in `/etc/systemd/system/` and
`/etc/systemd/user/`, udev rules `71-kraken-lcd`, `72-llama-view`,
`93-kraken-lcd-hidraw` and `94-llama-light-hidraw` in `/etc/udev/rules.d/`,
users from `/etc/sysusers.d/llama-bored.conf`. Stable device links:
`/dev/kraken-lcd/hid`, `/dev/llama-light/aura` and
`/dev/llama-light/keyboard`.

## Configuration

Four files in `/etc/llama-bored/`, one per service. The installer copies each
example from `packaging/` only if the file does not exist, and never
overwrites your edits. Every key has a default and a range (see the comments
in the examples), except `listen` and `allow` in `metrics.toml`, which are
required. Check a file without starting anything:

```sh
/usr/local/libexec/llama-bored/llama-light check --config /etc/llama-bored/light.toml
/usr/local/libexec/llama-bored/llama-metrics check --config /etc/llama-bored/metrics.toml
```

**`watch.toml`** (llama-watch):

| Key | Meaning |
|---|---|
| `[llama] enabled`, `url` | Poll llama-swap at a **loopback** URL (default `http://127.0.0.1:8080`). Point it at llama-swap itself, not a proxy in front of it |
| `[llama.backends]` | `"model-id" = "llamacpp"`, `"sglang"`, `"vllm"`, `"strata"` or `"openai"`: overrides the backend detected from the launch command |
| `[models.aliases]` | `"long-model-id" = "Short Name"` |
| `[load] cpu_limit_w`, `nominal_frac` | CPU full-load socket watts; `nominal_frac` (0.8) of each limit reads 100 |
| `[load] idle`, `gpu_idle_w`, `cpu_idle_w` | `"auto"` learns idle floors; `"fixed"` uses the watts as given |
| `[tty] blank_min`, `sleep_min` | Burn-in guard for the tty11 monitor: blank after N min without a keypress, power down at M (0 = off) |
| `[tty] show_text`, `prompt_view` | Show the live prompt/output (`true`) and strip chat templates (`"clean"`) or not (`"raw"`) |
| `[tty] gen_ceiling_tps`, `prompt_ceiling_tps` | Tops of the tok/s scales (250, 1500) |
| `[tty] palette` | `"llama"` (default): load llama-bored's 16 heat colours on tty11; `"vga"`: the kernel's colours |
| `[tty] font`, `size` | `"12x24"` (default) or `"12x22"`; `size = "COLSxROWS"` (160x26 to 1024x512) sizes tty11 at start. See below |
| `[tty] chart_glyphs`, `ctx_history_h` | `"eighths"` (bundled font) or `"halves"`; SLOTS history hours (1–24) |
| `[fans] enabled`, `hwmon`, `channels`, `labels` | Off by default. `hwmon` is a **name** from `cat /sys/class/hwmon/*/name` |

**`config.toml`** (kraken-lcd):

| Key | Meaning |
|---|---|
| `[upload] mode`, `stream_fps` | `"change"` (default) or `"stream"` (1–12 fps, 10 by default) |
| `[upload] min_interval_s` | Change-mode gap between uploads (floor 10). Re-run `install.sh` after changing it, so `RestartSec` follows |
| `[snapshot] stale_after_s`, `watch_down_stock_after_s` | When "no data" starts (1 s) and when the stock screen returns (30 s) |
| `[display] variant`, `rotate_deg` | `a1` (with tokens·24h) or `a3` (dial-first); 0/90/180/270 |

**`light.toml`** (llama-light):

```toml
[aura]
enabled = true
leds_per_fan = 6          # 1..=20
fans = "mirrored"         # "mirrored" (splitter) or "chain" (daisy chain)
# chain_len = 4           # fans = "chain" only, 1..=8
brightness_max = 80       # cap on every LED, 0..=100 %
fps = 10

[keyboard]
enabled = false           # Corsair STRAFE RGB MK.2, per key
brightness_max = 100

[[light]]                 # later entries draw over earlier ones
target = "aura.fans"      # or "aura.chain[2]", 'keyboard.keys["F1".."F12"]', "led:110"
metric = "activity"       # gpu, cpu, cpu_topk, load, mem, tokens_rate, coolant, gpu_temp, cpu_temp
range = [0, 100]
style = "solid"           # solid | ring | bar | pulse | ladder | gate | peak
palette = "act"           # act | thermal | mono, or stops = [[0, "#4A55C8"], [100, "#FF3A22"]]
smooth_s = 1.5
idle_color = "#101830"
```

With no `[[light]]` entries every fan shows activity on the LCD's colour ramp.
**Mirrored vs chained:** on a splitter every fan receives the same data, so
all fans always show the same thing. Per-fan metrics (`aura.chain[N]`) need
the fans daisy-chained on one data line (fan N is LEDs `N × leds_per_fan` on).
The example file has complete examples for both, and a full keyboard layout
(`keyboard-c`) with `[engine]` tweening, a `[base]` colour and named palettes.

**`metrics.toml`** (llama-metrics):

| Key | Meaning |
|---|---|
| `listen` | Required. `"0.0.0.0:19477"`; `"[::]:19477"` for dual-stack |
| `allow` | Required. CIDR allowlist, checked before a byte is read. Ships as `["192.168.0.0/16", "127.0.0.1/32"]`: **narrow it to your LAN** |
| `max_conns`, `stale_after_s` | Concurrent connections (16); snapshot age that counts as stale (5 s) |

Keep `allow` equal to `IPAddressAllow=` in `llama-metrics.service`, and the
port equal to `SocketBindAllow=`; the unit is the kernel's copy of the rule.
Use a drop-in (`IPAddressAllow=` on its own line first resets the list).
Enable it and open the port yourself:

```sh
sudo systemctl enable --now llama-metrics
curl -s http://127.0.0.1:19477/metrics | head
sudo firewall-cmd --permanent --zone=internal --add-port=19477/tcp && sudo firewall-cmd --reload
```

Scrape it from Prometheus:

```yaml
scrape_configs:
  - job_name: llamabored
    scrape_interval: 5s
    static_configs:
      - targets: ["ai-box.lan:19477"]
```

After editing `watch.toml` or `config.toml`, restart that service
(`llama-watch` or `kraken-lcd`). `light.toml` reloads by itself.

**tty11 on small or mixed screens.** Every display on a GPU shows the same
console, sized at boot for the largest one, so a smaller screen shows only
tty11's top-left corner. Size tty11 for the smallest screen:

```toml
[tty]
font = "10x18"     # "12x24" (default), "12x22" or "10x18": same glyphs
size = "192x60"    # COLSxROWS, 160x26..=1024x512; unset keeps the boot size
```

Which font suits which screen, at 1920x1080 (#73):

| Font | Console | Suits |
|---|---|---|
| `12x24` | 160x45 | a large screen at a distance |
| `12x22` | 160x49 | a 1080p screen at desk distance; FANS does not fit |
| `10x18` | 192x60 | a 1080p screen viewed up close: eight RECENT rows, and FANS (with TEMPS) beside IN/OUT |

From 190 columns FANS sits right of IN/OUT instead of under it.
`llama-watch.service` applies both before the watcher starts with a root
pre-step, `llama-watch tty-setup`, which runs `setfont` and `stty` from the
validated values (restart the unit after a change). Under 48 rows the
dashboard drops panels to fit: IN/OUT first, then FANS, then the chart
shrinks; it shows `tty too small` only below 160x26.

**TEMPS and FANS** (`watch.toml`, `[temps]` and `[fans]`, #74): TEMPS shows
every temperature the host exposes, found in `/sys/class/hwmon` (read only,
found again every 30 s, read at most once a second) plus the GPU, one row
per device with short labels, values coloured at the chip's own
`tempN_max` / `tempN_crit` (else 80 / 90 °C; 45 / 55 for a coolant):

```text
TEMPS  warn crit
CPU      Tctl 68 · CCD1 64 · CCD2 62
GPU      71
coolant  40
NVMe0    50 · s1 50 · s2 67
board    sys 51 · cpu 51 · aux0 27 · aux1 72 · aux2 16 · aux3 27 · aux4 50
NIC      51
Wi-Fi    39
```

Where it goes: from 190 columns, FANS then TEMPS right of IN/OUT; on a
narrower screen with rows to spare, FANS and TEMPS side by side under
IN/OUT; at 160x49 (12x22 at 1080p) TEMPS shrinks to one line of each row's
hottest value (`TEMPS  CPU 68 · GPU 71 · NVMe0 67 ...`) and FANS hides, so
OUT keeps its three rows. Use the 10x18 font for both panels whole.

Which sensors and fans show, the same way for both:

- `allow`: empty is everything discovered; otherwise only the matches.
- `block`: always hidden, whatever `allow` says.
- Built-in junk rules (`defaults = true`) apply to what no `allow` named,
  so `allow` can force an entry in. TEMPS hides the Super-I/O inputs
  that are unwired or repeat the CPU (`PCH_*`, `TSI*`, `SMBUSMASTER*`,
  `PECI*` on `nct*` / `w83*` chips) and any input that has not changed for
  ten minutes while another on its chip has. FANS hides a discovered fan
  that has never turned this run (an empty header); once it has spun it
  stays, so a stall shows.
- A reading below 5 °C or at 127 °C and above is never a temperature
  (unwired inputs read 0, -62 or 127) and is always dropped, even when
  allowed.

Patterns are globs (`*`, `?`; ASCII case ignored) over `chip` or
`chip:sensor`; a sensor matches by its label (`Composite`, `Pump speed`)
or its input (`temp3`, `fan2`). The chip is the hwmon `name` in lowercase
`[a-z0-9_.-]`; two chips of one name get a suffix from the device (the end
of the NVMe serial, `nvme-317k`; else the PCI address, `nvme-01.00`). List
them with `cat /sys/class/hwmon/*/name`. `rename` changes what tty11 shows
(the export keeps the real names):

```toml
[temps]
#enabled = true                    # on by default
#allow = ["k10temp", "nvme*", "nct6798:SYSTIN", "z53"]
block = ["nct6798:AUXTIN3"]
rename = { "nct6798:SYSTIN" = "board", "nvme-842p" = "scratch" }
warn = { "z53" = 42 }              # °C; the chip:sensor match wins over the chip
crit = { "z53" = 50 }

[fans]
enabled = true                     # off by default; then every fan on every chip
block = ["nct6798:fan7"]
rename = { "nct6798:fan2" = "rad top", "z53:fan1" = "pump" }
```

The older `hwmon = "nct6798"` + `channels = [2, 3, 5, 6]` (+ `labels`)
still work, as shorthand for `allow = ["nct6798:fan2", ...]` in that order
(those fans show even at 0 rpm); setting them together with `allow` is an
error.

**SETUP rules** (`watch.toml`, `[setup]`): each `[[setup.field]]` puts one
value on a SETUP row. Built-in rules
([`setup_defaults.toml`](crates/llama-watch/src/setup_defaults.toml)) cover
the upstream llama.cpp, vLLM and SGLang flags; rules in `watch.toml` are
added to them, and `defaults = false` keeps only yours.

```toml
[setup]
defaults = true

[[setup.field]]
row = "spec"              # the row it is drawn on, 1..=8 of A-Z a-z 0-9 _ - .
engines = ["vllm"]        # llamacpp, sglang, vllm, strata, openai; absent: any
match = "hyperqwen"       # only commands containing this (case ignored)
source = "env:SPEC"       # where the value comes from, below
kind = "token"            # how it is read, below
label = "method"          # text before the value: "method mtp"
suffix = ""               # text after it: " layers in RAM"
sep = " · "               # separator before this item (default " · ")
map = { mtp = "MTP" }     # value -> the whole item's text
default = "none"          # the item when there is no value (drawn grey)
fallback = false          # true: only when no earlier field of the row drew
group = "prefix"          # with fallback: only the fields of this group count
order = 40                # rows sort by their lowest order, items by theirs
```

Sources: `flag:-c,--ctx-size` (`--flag V`, `--flag=V`; the last one wins;
for a server, only after its entry point, or after the container image
when the command names no server but the engine is known from
`[llama.backends]` or the `/metrics` probe, #67), `env:NAME` (`-e NAME=V`,
`--env NAME=V`, `--env=NAME=V` or `NAME=V` in the wrapper before the
entry point or that image, or anywhere when neither is found),
`json:--speculative-config:num_speculative_tokens` (a key of a JSON
object given to a flag, one level deep), `live:NAME` (`engine`, `ctx`,
`kv_dtype`, `kv_block`, `prefix_cache`, `spec_accept`, `spec_len`,
`expert_hit`, `pcie_share`), `engine:KEY` (#54: a setting the engine
reports about itself, below), `name` (#54: the GGUF quant tag in the
llama-swap model `name`, `Flash Next Q4_K_M`, with kind `quant`; for an
engine whose command names no weights file), or no source at all for a
field that only shows its `default`. Kinds: `number` (drawn with thousands
separators), `token` (at most 16 characters of `A-Z a-z 0-9 _ . + -`; a
path-like value gives only its file stem), `quant` (the GGUF quant tag of
a model file name, `UD-Q4_K_M`), `present` (the flag is there: `on`),
`mib` (a number of MiB drawn as `16.4 GiB`, or `512 MiB` below 1 GiB).
An `engine:` source takes `number`, `token` or `mib` (default: as
reported).

`engine:KEY` keys are a fixed list the engine parser fills, each a plain
number, a short token or `on`/`off`; nothing else of an engine's report is
reachable. Strata (#54) fills, from `engine`: `engine`, `version`,
`context`, `max_context`, `kv`, `kv_resident`, `expert_slots`,
`expert_cache_mib`, `spec`, `mtp_max`, `lookup`, `spec_min_p`,
`pcie_frac`, `arena_mib`, `pool_workers`, `conversation_cache_slots`; from
`conversation_cache`: `conversation_cache` (`on`/`off`),
`conversation_cache_mib` (its budget, only while on),
`conversation_cache_requests`, `conversation_cache_requests_reused`,
`conversation_cache_reused_tokens`, `conversation_cache_prompt_tokens`,
`conversation_cache_evictions`; `requests` (`totals.requests`);
`max_tokens` (the request in flight); and of the newest finished request
`last_hit_rate`, `last_pcie_share`, `last_decode_tok_s`,
`last_drafts_offered`, `last_drafts_accepted`. The built-in rules show
Strata's rows above. Whatever a rule says, only numbers and short
tokens leave the launch command, never a path or a raw argument, and the
command itself is not kept. A bad rule (unknown kind or source, text too
long) fails `watch.toml` validation with its index:
`setup.field[3]: unknown kind "nubmer"`. `packaging/watch.example.toml`
has a copy-paste block for a vLLM container configured by `-e` env vars.

**tty11 colours** (`watch.toml`, `[tty]`): the console has 16 colour slots,
and llama-watch loads its own palette into them, built from the same heat
ramp as the LCD and the RGB (deep blue through violet and magenta to red,
blackbody gold for warnings, greys for text, black background):

```toml
[tty]
palette = "llama"  # default; "vga" keeps the kernel's VGA colours
```

The watcher sends the palette (`ESC ] P`) at start and with every full
redraw, and the unit's stop step (`llama-watch tty-reset`) puts the kernel's
colours back (`ESC ] R`), so a later login on tty11 is unaffected. Other
consoles are never touched. Set `palette` in `cast.toml` to the same value;
llama-view takes `--palette` (default `llama`).


## Watch tty11 over SSH or in tmux

`llama-view` mirrors tty11 read-only in any terminal. Its user needs the
`llama-view` group (`sudo usermod -aG llama-view $USER`, then log in again).

```sh
llama-view            # tty11, 10 fps; q or Ctrl-C quits
```

**Colours.** By default llama-view sends the exact colours tty11 shows, so
the terminal's colour theme does not repaint the dashboard:

- `--colors auto` (default): truecolor when `COLORTERM` is `truecolor` or
  `24bit`, else the nearest xterm-256 colour (indices 16-255, which themes
  leave alone).
- `--colors truecolor`: 24-bit SGR with tty11's RGB. SSH does not forward
  `COLORTERM`, so pass this (or set `LLAMA_VIEW_COLORS=truecolor`) when your
  terminal supports 24-bit colour.
- `--colors 256`, `--colors 16`: 256 colours, or the terminal's own 16
  (through its theme).
- `--palette llama|vga`: match `[tty] palette` in `watch.toml` (default
  `llama`).

`LLAMA_VIEW_COLORS` and `LLAMA_VIEW_PALETTE` set the same from the
environment, for saved SSH sessions that cannot pass flags; a flag wins.

**In tmux.** llama-view is made to live in a pane:

- It follows the pane's size and tty11's (which `[tty] size` or the font can
  change) every frame and repaints in full on any change. A pane smaller
  than tty11 shows the top-left corner (header, meters, RECENT) and a last
  line naming both sizes; `--fit center` shows the middle instead, and
  `--offset-x` / `--offset-y` pan.
- Each frame is one synchronized update (`ESC [ ? 2026 h`/`l`), so tmux 3.4+
  shows no tearing; only changed cells are sent.
- With focus events on, it drops to 1 frame a second while its pane is
  unfocused or the client is detached, and returns to `--fps` on focus.
- The pane title (`#T`) is `llama-view: <model> (<engine>) · <host>`, the
  model and engine read from tty11's header; the engine is its short name,
  left out when `<model> (<engine>)` would pass 40 characters. `--tmux-window-name` (or
  `LLAMA_VIEW_TMUX_WINDOW=1`) also names the tmux window after the model.
- It never rings the bell and never captures the mouse.

```tmux
# ~/.tmux.conf
set -g focus-events on                 # 1 fps while the pane is unfocused
set -as terminal-features ',*:RGB'     # tmux 3.2+: pass truecolor through
set -g allow-rename on                 # only for --tmux-window-name
set -g pane-border-status top          # optional: show #T on the pane border
```

Without the `RGB` feature tmux turns 24-bit colour into 256 colours itself
(`auto` picks 256 inside tmux anyway unless `COLORTERM` says truecolor).

`q`, `Ctrl-C` and `Ctrl-\` quit and restore the terminal. llama-view installs no
signal handlers (no `unsafe`): SIGWINCH is not needed (the size is polled),
and SIGTERM or SIGHUP end it without a restore. After a SIGTERM run `reset`,
or just start llama-view again: its exit turns canonical input and echo back
on whatever state it found.

## Watch tty11 on a TV

`llama-cast` shows the tty11 dashboard as a live video on DLNA/UPnP players
on your LAN. Tested with Roku Media Player on TCL Roku TVs. It announces a
media server over SSDP and streams 1920x1080 H.264 in MPEG-TS, rendered from
tty11 at 2 frames per second. Each viewer gets its own `ffmpeg` (libx264
required), two at a time by default.

The stream is tuned to start fast on a TV (#20). Each viewer's stream
begins with PAT, PMT and a keyframe, and is encoded for low latency (no
B-frames, no lookahead). The defaults below are in `cast.toml`:

| Key | Default | What it does |
|---|---|---|
| `bitrate_kbps` | `4000` | Constant bitrate, with filler on a still screen. A TV that waits for a fixed amount of data (Roku Media Player) starts in about a second instead of tens of seconds. `0` turns it off: least bandwidth, slowest start. 500..=20000. |
| `keyframe_s` | `1` | Seconds between keyframes, 1..=10. |
| `preroll_s` | `3` | Seconds of the first frame each new viewer gets at once, 0..=10. The picture then runs this far behind tty11. |

Each viewer uses `bitrate_kbps` of LAN bandwidth. If the TV still waits,
raise `preroll_s` or `bitrate_kbps`.

It is LAN-exposed by design: any host in `allow` can watch whatever tty11
shows. It reads tty11's screen read-only, writes no file, and connects to
nothing. The installer installs it but never enables it.

1. Edit `/etc/llama-bored/cast.toml`: set `listen` and `interface_addr` to
   this host's LAN address, narrow `allow` to your LAN or to the TVs, and
   set `font` and `palette` to the same values as `[tty] font` and
   `[tty] palette` in `watch.toml`.
2. Keep `llama-cast.service` in step with the config. `IPAddressAllow=`
   must be `allow` plus `239.255.255.250/32`, and `SocketBindAllow=` must be
   `tcp:<listen port>` and `udp:1900`. Use a drop-in, as the unit's comments
   show.
3. Check the config, then enable the unit:

   ```sh
   sudo /usr/local/libexec/llama-bored/llama-cast check --config /etc/llama-bored/cast.toml
   sudo systemctl enable --now llama-cast
   ```

4. Open SSDP and the stream port to your LAN in the firewall, for example:

   ```sh
   sudo firewall-cmd --permanent --zone=internal --add-port=1900/udp --add-port=19478/tcp
   sudo firewall-cmd --reload
   ```

5. On the TV, open Roku Media Player, choose **llama-bored (HOST)**, then
   **tty11 on HOST**, HOST being this machine's host name.

The TV lists each server by its name, so the default `name` carries the
host name: two servers, or one and a stale entry the TV kept for it, are
told apart (#21). `name` and `title` (the video's title) in `cast.toml`
replace `{host}` with the host name's first label (letters, digits, `-`
and `_`, at most 32 characters); without `{host}` they are used as
written. A `cast.toml` from an earlier release that sets `name = "llama-bored"`
keeps that name; change it to `"llama-bored ({host})"` or remove the line.
The device id the TV keys an entry on is a hash of `/etc/machine-id`:
stable on this host and different on another, and it does not reveal the
machine id. llama-cast says goodbye over SSDP when it starts and twice
when it stops, so a TV drops an old entry sooner.

## Troubleshooting

| Symptom | What to check |
|---|---|
| "no NZXT Kraken Z LCD (1e71:3008) found" | `lsusb -d 1e71:`. Other NZXT models are not supported |
| kraken-lcd exits: no `z53` hwmon | The `nzxt-kraken3` driver is not bound: `cat /sys/class/hwmon/*/name` |
| `install.sh` refuses the tree | Commit or stash, `git switch main`, re-run `scripts/check.sh`, install without editing |
| `install.sh`: "a previous install failed" | Run the rollback in `ROLLBACK_PENDING`, then retry |
| LCD says **no data** | llama-watch is stopped or failing: `journalctl -u llama-watch` |
| LCD says **AI down** | llama-swap is not answering at `[llama] url` |
| LCD frozen, `journalctl -u kraken-lcd -p crit` shows `cooling guard halted` or `cooling guard latch is present` | The cooling guard tripped. Check the cooler, read `/var/lib/kraken-lcd/halted`, then `sudo /usr/local/libexec/llama-bored/kraken-lcd clear-halt` and restart kraken-lcd |
| `show-image`: interface 0 is bound | kraken-lcd is running. Stop it first |
| LCD or RGB stops after a replug or resume | The hidraw node got a new number; restart `kraken-lcd` or `llama-light` |
| Gauge never passes ~50, or pegs at idle | Tune `[load] cpu_limit_w` and the idle watts; the ACTIVITY row shows the source |
| tty11 IN/OUT empty | `LLAMA_SERVER_SLOTS_DEBUG=1` for llama-server, and `[tty] show_text` |
| tty11 chart shows odd glyphs | The font did not load; set `chart_glyphs = "halves"` |
| RGB does nothing | `lsusb -d 0b05:18f3`, `ls -l /dev/llama-light/aura`, `journalctl -u llama-light` (`aura absent (...)` names why), `/usr/local/libexec/llama-bored/llama-light check --config /etc/llama-bored/light.toml`. Only one RGB tool (OpenRGB, vendor tools) at a time |
| Keyboard stays on its own lighting | `lsusb -d 1b1c:1b48`, `ls -l /dev/llama-light/keyboard`, `[keyboard] enabled = true`. Only one RGB tool (ckb-next, OpenRGB) at a time. Plugged in after llama-light started: it restarts itself to pick it up |
| All fans show one colour with `aura.chain[N]` | The fans are on a splitter. Use `fans = "mirrored"` or daisy-chain them |
| Prometheus gets no answer | `allow`, `IPAddressAllow=`, and the firewall; `llamabored_exporter_rejected_connections_total` counts refusals |
| `llamabored_snapshot_stale` is 1 | llama-watch is down or behind |
| `llama-view`: permission denied | Join `llama-view`, then log in again |
| LCD stays on stock | Another tool (CoolerControl, liquidctl scripts) may drive the LCD. Run one |

## Uninstall and rollback

**One install run:** `install.sh` prints the exact commands that undo it. A
failed run also saves them to `/usr/local/libexec/llama-bored/ROLLBACK_PENDING`,
and the next install refuses until you run them. Stop the writers first.

**Full uninstall:**

```sh
sudo systemctl disable --now llama-metrics llama-light kraken-lcd llama-watch
sudo systemctl --global disable kraken-lcd-halt.path
sudo rm -rf /usr/local/libexec/llama-bored /usr/local/share/llama-bored /etc/llama-bored /var/lib/kraken-lcd
sudo rm -f /usr/local/bin/llama-view /etc/sysusers.d/llama-bored.conf \
  /etc/systemd/system/{kraken-lcd,llama-watch,llama-light,llama-metrics}.service \
  /etc/systemd/user/kraken-lcd-halt.path /etc/systemd/user/kraken-lcd-halt-notify.service \
  /etc/udev/rules.d/{71-kraken-lcd,72-llama-view,93-kraken-lcd-hidraw,94-llama-light-hidraw}.rules
sudo rm -rf /etc/systemd/system/llama-metrics.service.d   # the IPAddressAllow= drop-in, if you made one
sudo systemctl daemon-reload && sudo udevadm control --reload
sudo udevadm trigger --action=change --subsystem-match=hidraw
sudo udevadm trigger --action=change --attr-match=idVendor=1e71 --attr-match=idProduct=3008
for u in kraken-lcd llama-watch llama-light llama-metrics llama-cast; do sudo userdel "$u"; done; sudo groupdel llama-view
```

Close the firewall port if you opened it. Nothing on the devices needs
undoing: the Kraken's frame memory and the Aura direct colours are RAM, and
the board's stored lighting effect returns at the next power cycle.

## Credits and license

Built with Claude Code and Grok: Claude (Opus, Fable) for design, review,
security and much of the code; Grok CLI for many of the builds.

MIT; see [LICENSE](LICENSE). The console
font is derived from Hack (MIT, with Bitstream Vera portions) and the LCD
renderer bundles Inter (SIL OFL 1.1); see [NOTICE](NOTICE). Not affiliated
with NZXT, ASUS or Corsair.
