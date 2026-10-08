# Changelog

All notable changes to llama-bored. Versions follow
[Semantic Versioning](https://semver.org/); the process is in
[docs/RELEASING.md](docs/RELEASING.md).

## Unreleased

- The snapshot size cap is 32 KiB, up from 16 KiB (#84), so per-model KV, counters, temperatures and fans fit with room to spare. Every reader takes the cap from llama-core; after upgrading, restart every unit so no older reader meets a larger snapshot.

- RECENT's context-bar colours run red (smallest step) through green to blue (largest) instead of violet to orange, and the header key follows (#83).

Ships as 0.5.0: the exporter's series change incompatibly (#71).

### Upgrading

- Dashboards and alerts on llama-metrics need a migration: every per-model series is now labelled `model` (the llama-swap id) and `engine` instead of `name` and `full_name`, and many names changed. Use the table below. Install all binaries together (`install.sh`) and restart `llama-watch` and `llama-metrics`; the wire stays schema 1, so the LCD and RGB keep working through the restart.
- `llamabored_fan_rpm` and `llamabored_fan_pwm_ratio` gain a `chip` label (#74), since fans now come from any hwmon chip; queries by `label` or `channel` keep working, but two chips can both have channel 1. `[fans]` with `hwmon` + `channels` keeps its fans and order.

### Changed (breaking)

- llama-metrics exports one normalized series per quantity for every engine (#71), told apart by `model` and `engine` (`llamacpp`, `vllm`, `sglang`, `strata`, `openai`); descriptive strings move to `llamabored_model_info`; names carry units, ratios are 0 to 1, only counters end in `_total`; window means and rates give way to counters for `rate()`. README (**llama-metrics** under Features) has the full list and PromQL. Old to new:
- `llamabored_model_kv_cache_usage_ratio` and the snapshot's `kv_fill` are KV in use / capacity across all sessions where the engine reports tokens (#79), else the engine's own ratio. SGLang's no longer comes from `token_usage`, which is the fullest of its full-attention, sliding-window and Mamba pools (a hybrid model read near full), but from its KV token gauges, else `full_token_usage`; llama.cpp (b11429 on, which dropped `kv_cache_usage_ratio`) and Strata now have it
- llama-metrics exports one normalized series per quantity for every engine (#71), told apart by `model` and `engine` (`llamacpp`, `vllm`, `sglang`, `strata`, `openai`); descriptive strings move to `llamabored_model_info`; names carry units, ratios are 0 to 1, only counters end in `_total`; window means and rates give way to counters for `rate()`. README "Metrics" has the full list and PromQL. Old to new:

| Old | New |
|---|---|
| `llamabored_activity_pct` | `llamabored_activity_ratio` (÷ 100, 0 to 1.25) |
| `llamabored_load_pct` | `llamabored_load_ratio` |
| `llamabored_cpu_pct` | `llamabored_cpu_utilization_ratio` |
| `llamabored_cpu_topk_pct` | `llamabored_cpu_topk_utilization_ratio` |
| `llamabored_gpu_pct` | `llamabored_gpu_utilization_ratio` |
| `llamabored_mem_pct` | dropped: `llamabored_memory_used_bytes / llamabored_memory_total_bytes` |
| `llamabored_tokens_decoded_total` | dropped: `sum(llamabored_model_generation_tokens_total)` |
| `llamabored_tokens_prompt_total` | dropped: `sum(llamabored_model_prompt_tokens_total)` |
| `llamabored_model_loaded{name, full_name, quant, kv, ctx, moe, backend}` | `llamabored_model_info{model, engine, display_name, quant, kv_type, version}`; `ctx` is `llamabored_model_context_size_tokens`; `moe` is dropped |
| `llamabored_model_state{name, full_name, state}` | `llamabored_model_state{model, engine, state}` |
| `llamabored_model_ctx_size_tokens` | `llamabored_model_context_size_tokens` |
| `llamabored_model_requests_running` | same name; llama.cpp now fills it too (`requests_processing`) |
| `llamabored_model_requests_queued` | `llamabored_model_requests_waiting` (llama.cpp: `requests_deferred`) |
| `llamabored_slots_busy` | `llamabored_model_requests_running` |
| `llamabored_slots_total` | `llamabored_model_slots` (vLLM, SGLang and Strata: their request cap) |
| `llamabored_model_cache_hit_ratio` | dropped: `rate(llamabored_model_prompt_cached_tokens_total[5m]) / rate(llamabored_model_prompt_tokens_total[5m])` |
| `llamabored_model_spec_acceptance_ratio` | dropped: `rate(…_spec_accepted_tokens_total[5m]) / rate(…_spec_draft_tokens_total[5m])`; SGLang, which reports only a gauge, has none |
| `llamabored_model_spec_accepted_length` | dropped: `1 + rate(…_spec_accepted_tokens_total[5m]) / rate(…_spec_drafts_total[5m])` (vLLM) |
| `llamabored_model_ttft_seconds` | `llamabored_model_time_to_first_token_seconds` summary: `rate(_sum) / rate(_count)` |
| `llamabored_model_itl_seconds` | `llamabored_model_inter_token_latency_seconds` summary |
| `llamabored_model_e2e_latency_seconds` | `llamabored_model_request_duration_seconds` summary (every engine) |
| `llamabored_model_prefill_tokens_per_second` | `rate(prompt_tokens_total − prompt_cached_tokens_total) / rate(llamabored_model_prefill_seconds_total)` |
| `llamabored_model_decode_tokens_per_second` | `rate(llamabored_model_generation_tokens_total) / rate(llamabored_model_decode_seconds_total)` |
| `llamabored_slot_ctx_used_tokens{name, full_name, slot}` | `llamabored_slot_context_used_tokens{model, engine, slot}` |
| `llamabored_slot_ctx_resets_total{…, reason}` | `llamabored_slot_context_resets_total{model, engine, slot, reason}` |
| `llamabored_model_prompt_tokens_total`, `…_prompt_cached_tokens_total`, `…_spec_drafts_total`, `…_spec_draft_tokens_total`, `…_spec_accepted_tokens_total`, `…_preemptions_total`, `…_sleeping`, `…_kv_cache_usage_ratio`, `…_kv_block_size_tokens`, `…_prefix_caching`, `…_expert_cache_hit_ratio`, `…_pcie_share_ratio` | same names, labels `model` and `engine` |

### Added

- KV cache in use per model, across all sessions, and its capacity (#79). tty11 shows it as one item on each model's SLOTS engine line (`KV 54k/204k tok 27 % · +31k cached`; `≈` for vLLM's ratio-derived count, `shared` for a unified llama.cpp cache, `· 2 sessions` for slot engines), and a llama.cpp model gets a `llamacpp` line under its slot rows. llama.cpp: the sum of every slot's held `/slots` tokens, idle slots included; capacity one slot's `n_ctx` with a unified cache (`--kv-unified`, or no `-np`), else the sum, told from the launch command (assumed from equal `n_ctx` when it cannot be read). vLLM: `kv_cache_size_tokens` (else blocks × block size) as capacity, its usage ratio × that in use. SGLang: `kv_used_tokens`, `kv_evictable_tokens`, `max_total_num_tokens` (they lag up to 40 decode steps). Strata: `context` × `batch_slots`, and its `live.slots` tokens. New gauges `llamabored_model_kv_used_tokens`, `…_kv_capacity_tokens`, `…_kv_cached_tokens`, `…_kv_sessions`, `…_kv_unified`; vLLM's capacity in tokens and blocks and its max concurrency are `[setup]` keys `engine:kv_tokens`, `engine:kv_blocks`, `engine:kv_max_concurrency`. Snapshot (schema 1, additive): each model's `kv` object `{u, t, c, s, h, a}` (used, capacity, cached, sessions, shared pool, approximate), tokens at most 2^32 − 1; the worst-case snapshot grows from 15,525 to 16,165 bytes of the 16,384 cap. `≈` joins the bundled console fonts (appended, no glyph moves) and tty11's glyph allowlist
- tty11 RECENT: a rainbow key in the header row (#77), the eight remainder colours as full blocks in order, left-aligned directly over the context bar's first cells (same palette slots on tty11, llama-view 16 / 256 / truecolor and llama-cast; full blocks in halves mode too; a narrower bar shows what fits), so a remainder cell reads as the start, middle or end of its eighth at a glance. It replaces the legend's small rainbow strip and its "per 1/8"; `█cached █new █out` stays
- tty11 RECENT: the bar shows each request's context instead of its speed (#75), cached / new input / output as three runs, linear against the model's context size (remembered after it unloads), each run led by a remainder cell whose height is the eighths and whose colour (a fixed eight-slot order, the same on tty11, llama-view and llama-cast) the step within the eighth; a yellow `!` at 90 % of the context; `~` and the largest row as the scale when the size is unknown; `▄`/`█` with `chart_glyphs = "halves"`. The legend gains the colour key and the rainbow order where it fits
- tty11 RECENT in-flight rows (#75): the request running now leads RECENT, from llama.cpp's `/slots` (each busy slot, at most half the rows) or Strata's `live`, marked `pp` (with the reset letter after a compaction, new conversation or eviction) or `gen`, its prompt filling the bar poll by poll; llama-swap's finished row replaces it, never both. vLLM, SGLang and OpenAI-compatible servers keep the old `gen` mark on the newest finished row
- tty11 TEMPS panel and `llamabored_temperature_celsius{chip, sensor}` (#74): every temperature in `/sys/class/hwmon` (read only; found again every 30 s, read at most once a second) plus the GPU, grouped by device with short labels (`CPU  Tctl 68 · CCD1 64 · CCD2 62`, `NVMe0  50 · s1 50 · s2 67`), coloured at each chip's own `tempN_max` / `tempN_crit` (else 80 / 90 °C, 45 / 55 for a coolant). Beside IN/OUT under FANS from 190 columns, side by side with FANS under IN/OUT on taller narrow screens, one line of hottest values at 160x49. Two chips of one name get a device suffix (`nvme-317k`). Unreadable inputs (`EIO`, `ENODATA`) are skipped with one log line per change. Readings below 5 °C or from 127 °C are always dropped; built-in rules hide the Super-I/O `PCH_*`, `TSI*`, `SMBUSMASTER*` and `PECI*` inputs and an input that stays unchanged for ten minutes while its chip's others move. New `[temps]` in watch.toml: `enabled` (default true), `allow`, `block`, `defaults`, `rename`, `warn`, `crit`
- `[fans]` discovers every fan on every hwmon chip when `hwmon` is unset (the AIO's pump and fan next to the board's), hiding a header that has never turned this run, and takes the same `allow` / `block` / `rename` / `defaults` as `[temps]` (#74). `hwmon` + `channels` (+ `labels`) still work as shorthand for `allow = ["<hwmon>:fanN", ...]`; setting them together with `allow` is refused
- Snapshot (schema 1, additive): `temps` (at most 32 `{c, s, t}` rows, tenths of a degree) and `fan_rows` (at most 16 `{c, n, l, r, p}` rows with their chip). The watcher sends fans as `fan_rows` and no longer as `fans`, since two chips may both have a `fan1` and an older reader refuses a repeated channel; an older llama-metrics then shows no fans until it is upgraded too. A snapshot that would pass the 16 KiB cap (only the pathological eight-models-at-their-widest case does) drops temperature rows from the end, then fan rows
- tty11: a smaller bundled console font, `llama-hack-10x18.psfu` (`[tty] font = "10x18"`, also in cast.toml), Hack at 15 px built by `build-psf.py` and checked byte for byte like the others (#73). On a 1920x1080 screen it gives 192x60 instead of 160x49 with 12x22: eight RECENT rows, and from 190 columns (was 200) FANS sits beside IN/OUT. README has a font-per-screen table. Its meter glyph `▇` leaves a 2-pixel gap (3 in 12x22 and 12x24), still an eighth of the cell
- New per-model series (#71): `llamabored_model_generation_tokens_total` (every engine; llama.cpp's `tokens_predicted_total`, an OpenAI-compatible server's activity rows), `llamabored_model_prefill_seconds_total` and `…_decode_seconds_total` (llama.cpp, vLLM, Strata), `llamabored_model_requests_total{status="ok"|"error"}` (llama-swap's activity rows), the TTFT and ITL summaries (vLLM, SGLang), request duration for every engine (engine e2e histograms, else activity durations), llama.cpp's speculative draft and accepted tokens (activity draft fields), `llamabored_model_slots` and `llamabored_model_info{version}` (Strata). Every counter counts since llama-watch started and stays monotonic across engine restarts and unloads within the run
- Snapshot (schema 1, additive; older readers ignore the new keys): each model's llama-swap `id`, engine `version`, `max_running`, and a `counters` object (`gen_tokens`, `prefill_ms`, `decode_ms`, `req_ok`, `req_err`, `ttft` / `itl` / `e2e` as `{ms, n}`), each at most 2^53 − 1. `running`, `queued`, `kv_fill` and the speculative token counters are filled for llama.cpp too. No longer sent, since nothing but the old exporter read them: `cache_hit`, `slots_busy` and the engine window means (`spec_len`, `ttft_s`, `itl_s`, `e2e_s`, `prefill_tps`, `decode_tps`); tty11 still shows them. The worst-case snapshot is 15.5 KiB of the 16 KiB cap
- Self-check for suspected loads (#70). `/running` is read again right after each upstream read; a model `starting` there, or an upstream read slower than 2 s, is a suspected load: a warning names the model id and path, that model gets no upstream reads for 5 minutes, and it is counted. The counter rides the snapshot as the optional `suspected_loads` list (`model`, `count`; at most 8 rows, additive on schema 1) and llama-metrics exports it as `llamabored_collector_suspected_loads_total{model}`, absent while there are none
- llama-metrics test: a snapshot with no model running is a 200 with valid exposition, `llamabored_ai_state` and no model series (#70)
- Snapshot (schema 1, additive; older readers ignore the new keys): each model's llama-swap `id`, engine `version`, `max_running`, and a `counters` object (`gen_tokens`, `prefill_ms`, `decode_ms`, `req_ok`, `req_err`, `ttft` / `itl` / `e2e` as `{ms, n}`), each at most 2^53 − 1. `running`, `queued`, `kv_fill` and the speculative token counters are filled for llama.cpp too. No longer sent, since nothing but the old exporter read them: `cache_hit`, `slots_busy` and the engine window means (`spec_len`, `ttft_s`, `itl_s`, `e2e_s`, `prefill_tps`, `decode_tps`); tty11 still shows them. The worst-case snapshot is 15.5 KiB of the 16 KiB cap (15.8 KiB with #79's `kv`)

### Fixed

- RECENT and SLOTS read llama-swap's `input_tokens` per engine (#82): it leaves the cached tokens out for llama.cpp and Strata (`timings`) and counts them for vLLM, SGLang and OpenAI `usage`. One helper makes it the whole prompt and its cached part for RECENT's IN and CACHED, the context bar, the per-model prompt counters fed from activity rows (a vLLM or SGLang model counted from its activity rows no longer adds its cached tokens twice to `llamabored_model_prompt_tokens_total`) and the in-flight and reset matching, so cached llama.cpp and Strata rows no longer under-draw. SLOTS counts a slot's computed prompt tokens against the exact prompt once it decodes (bar full) and against what it holds so far, `8,192/69,632+`, in prefill (no bar). A new task whose reused prefix is far below what its slot held is a drop even when its prefill finished before the first poll that saw it; the sparkline marks it the same way. README's example shows the #78 prefill row
- llama-watch's polling can no longer make llama-swap load or swap a model (#70). `/running` was read every 0.5 s and `/upstream/<id>/metrics` every 0.25 s, so a read could act on a ready list up to 0.5 s old: if a client asked for another model in that window, our upstream read made llama-swap load ours back. Every upstream read (`/metrics`, `/slots`, the engine probe) now goes through one function, the fresh gate: it reads `/running` as the request just before the GET, and sends it only for a model that read lists as `ready`; a model that is not is dropped, not retried. If that read lists any model as `starting`, `stopping` or in another state, a swap is in progress and the round makes no upstream request at all. A `409` (llama-swap's `upstream.ignorePaths` answer for a model that is not loaded) or any other non-2xx from an upstream read means "not available now": the model's engine numbers are dropped as on unload, it gets no second read that round, llama-swap stays up, and a 409 is logged once per change (`<id>: llama-swap says not loaded; skipping until ready`) and counts as no tap failure. Redirects are still never followed, now pinned by S17. Cost: two small `/running` reads per upstream read
- README: "Never swaps models" under "Backends behind llama-swap", with llama-swap's `upstream.ignorePaths` block (the default pattern kept, since listing `ignorePaths` replaces it) as optional extra protection for the short window that remains between our `/running` read and our upstream read (#70)
- llama.cpp in-flight rows, SLOTS context and reset reasons read `/slots` `n_prompt_tokens` right (#78). On current llama-server (b11429) it is the tokens the slot holds, not the whole prompt: cached + processed so far in prefill, and prompt + output (all but the newest token) while decoding; `/slots` sends no prompt length at all. A RECENT in-flight row in prefill now shows what is held, IN as a lower bound (`69,632+`) with no target track (it used to grow its own target); once it decodes the prompt is exact (cached + processed) and IN no longer counts the output. Its llama-swap row (input + cache) replaces it at once. The ctx fill, sparkline and held context no longer count the output twice. A context drop first seen in prefill waits for the decode's whole prompt before it matches an activity row or takes a cached share, and is `unknown` if its task is never seen decoding. PROMPT `last` shows the whole prompt
- README: "Never swaps models" under the engines, with llama-swap's `upstream.ignorePaths` block (the default pattern kept, since listing `ignorePaths` replaces it) as optional extra protection for the short window that remains between our `/running` read and our upstream read (#70)

### Added

- Self-check for suspected loads (#70). `/running` is read again right after each upstream read; a model `starting` there, or an upstream read slower than 2 s, is a suspected load: a warning names the model id and path, that model gets no upstream reads for 5 minutes, and it is counted. The counter rides the snapshot as the optional `suspected_loads` list (`model`, `count`; at most 8 rows, additive on schema 1) and llama-metrics exports it as `llamabored_collector_suspected_loads_total{model}`, absent while there are none
- llama-metrics test: a snapshot with no model running is a 200 with valid exposition, `llamabored_ai_state` and no model series (#70)

## 0.4.1 — 2026-10-06

### Fixed

- A llama.cpp model whose `/slots` carries no prompt or generated text now gets IN and OUT from llama-swap's request captures (#66). Current llama-server (b11429) sends `prompt` and `generated` only with `LLAMA_SERVER_SLOTS_DEBUG=1`, which an env var on llama-swap does not pass into a server's own container, so IN and OUT stayed empty with no reason given. A model whose newest `/slots` body with a task has neither key (any slot with one counts as text) gets the last finished exchange from its captures, as vLLM and SGLang do: the same `GET /api/captures/<id>`, once per new row, text on and `has_capture` only, same cap; no new path. It is logged once per model: `<id>: /slots has no prompt text; start llama-server with LLAMA_SERVER_SLOTS_DEBUG=1 for live IN/OUT`. Every `/slots` read checks again, so text that appears (the server restarted with the variable) switches IN and OUT back to live and drops the capture. SLOTS, ctx fill, prompt progress and reset reasons still come from `/slots`, whose `n_prompt_tokens` and `n_prompt_tokens_cache` are still sent; the older shape with text always present reads as before. `CaptureView` gains `over_slots`
- A llama.cpp server started as a container image (`podman run … <image@digest> --host … -m …/X-IQ4_XS.gguf -c 262144 -ctk f16 -ctv f16 -ncmoe 39 …`) now gets its ctx, KV, `-ncmoe` and quant in SETUP and on the exporter (#67). The flag parser ran only on a command that names `llama-server`, so a model told by its `/metrics` (#31) showed `quant="" kv="" ctx="" moe=""`. Once the engine is known from the command, the probe or `[llama.backends]`, that engine's flags are read from the arguments after the image of a `podman run` / `docker run` (also `container run`): the image is the first word after `run` that is neither an option nor an option's value (a bare digest or image id, `name@sha256:…`, `registry/name:tag`), every long option takes a value unless it is one of the known boolean ones (`--rm`, `--init`, `--privileged`, …) or carries `=`, and `-it`-style short clusters are read as podman does. With no image found the whole command is read, as before. A probe that names a server re-reads `/running` at once, so the detail does not wait for the next poll. A command that names its server is read after its entry point for llama.cpp too, so a wrapper's own `-c` or `-m` lends nothing. The `[setup]` rules read server flags after the image and env before it. Same allowlist: numbers and short tokens only, the command is not kept. Fixture: an invented llama.cpp image entry in `fixtures/llama/running-setup.json`

## 0.4.0 — 2026-10-05

### Upgrading

- Run `install.sh`, then restart every unit: `systemctl try-restart llama-watch kraken-lcd llama-light llama-metrics llama-cast`. The kraken-lcd unit gains `ExecStop=` and `RuntimeDirectory=kraken-lcd` (#59); the first stop after the upgrade still uses SIGTERM.
- `cast.toml`: a file that sets `name = "llama-bored"` keeps that exact name. Change it to `"llama-bored ({host})"`, or remove the line, so the TV tells servers apart (#21). The new keys `bitrate_kbps`, `keyframe_s` and `preroll_s` default to the fast start (#20); set `bitrate_kbps = 0` and `preroll_s = 0` for the lean stream.
- tty11's header clock and RECENT now show local time, not UTC (#48).

### Changed

- llama-cast servers are told apart in the TV's list (#21). The friendly name defaults to `llama-bored ({host})` and the video's title to `tty11 on {host}`; `{host}` is the kernel host name (`uname(2)`, no file read): its first label, letters, digits, `-` and `_` only, at most 32 characters, else `host-` and 8 hex digits of the UDN hash. New `cast.toml` key `title`; `name` and `title` are 1..=64 printable characters with braces only in `{host}`, and the expanded text is cut to 64. A `cast.toml` that sets `name = "llama-bored"` keeps that exact name. The UDN stays the SHA-256 hash of `/etc/machine-id` (stable per host, not the id). SSDP: one `ssdp:byebye` round at start, so a TV drops an entry cached from before a restart, and two rounds 100 ms apart on stop (`llama-cast bye`, the unit's `ExecStop=`). `llama-cast check` prints the friendly name and title it would use

### Fixed

- llama-cast starts much sooner on a Roku (#20). A still dashboard encoded to about 70 kbit/s (2 fps in, `-tune stillimage`, quality-based x264, B-frames and lookahead), so Roku Media Player, which waits for a fixed amount of data before it plays, sat at its loading percentage for tens of seconds. The stream is now constant bitrate with filler (`bitrate_kbps`, default 4000; `nal-hrd=cbr`), each new viewer gets `preroll_s` (default 3) seconds of the first frame at once before frames are paced, and x264 runs `stillimage,zerolatency` with no B-frames and a `keyframe_s` (default 1) keyframe interval; ffmpeg flushes packets as muxed and repeats PAT/PMT every 100 ms. Each viewer's stream already began with PAT, PMT and an IDR (one ffmpeg per viewer); a test now checks it. The stream response also sends `contentFeatures.dlna.org` with the item's profile, `DLNA.ORG_OP=00` and flags. New `cast.toml` keys `bitrate_kbps` (0 or 500..=20000), `keyframe_s` (1..=10) and `preroll_s` (0..=10); an existing `cast.toml` gets the defaults. S18's pinned argument vector is the new one, still built only from validated numbers
### Fixed

- kraken-lcd: a stop or restart no longer cuts an upload in half (#59). In stream mode the writer is almost always uploading, and a SIGTERM between `WriteStart` and `WriteEnd` left the Kraken refusing `DeleteBucket` for over a minute, so the next process spent its 60 s startup grace and `fail_limit`, restored stock and exited 1. The unit's new `ExecStop=` creates `/run/kraken-lcd/stop` (new `RuntimeDirectory=kraken-lcd`, mode 0700, empty at each start) and waits up to 5 s for the main process; the writer checks the flag at the top of every tick, so the upload in flight runs to its end, no new one starts, and it exits 0 (`stop requested; no upload in flight`). SIGTERM stays the backstop and `ExecStopPost` still restores stock. The writer only reads the flag; no signal handler, no `unsafe`, no new dependency. The protocol has no command to reset a bucket the device refuses to delete, so start-up recovery is unchanged (`ExecStopPost`'s `ShowLiquid`, slot hygiene at open, the #14 grace)
- The S1 write-API source scan now sees fully qualified and free-function writes (#23): `Write::write*` / `io::Write::` / `std::io::Write::` / `<T as Write>::` calls and function values, `fs::write`, `fs::copy`, `io::copy`, `rustix::io::write` / `pwrite` / `writev` / `pwritev` (called, imported, renamed or in a `{...}` import group), `.write_fmt(` and `.write_vectored(`, however spaced or wrapped; `fmt::Write`'s in-memory `write_str` / `write_char` are not writes. Four writers it had missed are allowlisted with a reason each: llama-watch `tty/term.rs` (tty11) and `publish.rs` (the snapshot temp file), llama-view `main.rs` and `render.rs` (its own terminal); llama-view `main.rs` names `/proc/sys/kernel/hostname` only to read it and is waived from the file-level S3 rule, the line-window S3 check still applies. Self-tests plant each form, check the look-alikes stay clean, and check every newly listed file is still seen as a writer
### Fixed

- install.sh no longer piles up `config.toml.bak-<sha>` files (#57). The writer config is backed up only when the install actually changes it (a `config.toml.new` that differs from the live file); a kept or identical config is neither written nor backed up. After a successful install, each file it manages (binaries, units, rules, fonts, `INSTALLED_SHA`, `config.toml`, `watch.toml`, `light.toml`, `metrics.toml`, `cast.toml`) keeps only its newest three `<file>.bak-<40 hex>` / `.bak-none` backups by mtime (the other configs are only written when absent, so install.sh makes no new backups of them, but old ones are trimmed too); the backup this run's rollback names is always kept. Hand-named backups such as `watch.toml.bak-pre-setup`, symlinks and other files' backups are never touched. Self-test: no backup for an unchanged config, pruning to three, hand-named backups kept
- tty11 RECENT's TIME is in the host's local zone (#48). llama-swap's activity `timestamp` is parsed as RFC 3339 (`Z` or `±hh:mm`, optional fraction) and shown as `YYYY-MM-DD HH:MM:SS` local time; text that does not parse is shown sanitised as before. The header clock and the AI-down label were UTC too and now use the same zone. No zone crate and no libc: a small TZif reader (v1 and v2+ blocks, the footer POSIX rule) loads `TZ` or `/etc/localtime` (readable under the unit's `ProtectSystem=strict`), falling back to UTC, and rereads it at most once a minute. Sorting and the row fingerprint still use llama-swap's raw text. Fixture: `fixtures/tz/America-New_York.tzif` (tzdata, public domain)

## 0.3.6 — 2026-10-05

### Added

- LCD current tok/s (#55): the generation rate over the tokens chart's far (24 h) end, 24 px Inter ExtraBold right-aligned and centred in the plot, with a 2 px black knockout and a soft dark backing so it reads over a bright fill; the unit `tok/s` replaces the `tokens · 24h` caption in the title row above it, off the data. A 5 s moving average of decoded tokens (τ = 5 s, from the counter intervals), reset by a gap (first reading, `run_id` change, stall, counter going down or missing); 3 s without a decoded token (prefill, idle) snaps it to `0` instead of leaving a decaying tail. `7.5` under 10, `112` up to 999, `1.2k`, `12k`; idle `0` and no counter or no data `—` in #666. Change mode holds it in its printed step (0.1, 1, 100 tok/s, then 1k) with a 30 % margin; stream mode shows it every frame. View (frame key, additive): `gen_tps_tenths`. Goldens: every A1 `layout_a` frame (caption and numeral); the `a1-v3b-*` and `a3-v3b-pinned` view fixtures gain a rate
- Strata's `/metrics` read in full (#54). Speculative decoding from `totals.drafts_offered` / `drafts_accepted` (acceptance over the window, draft and accepted token counters; Strata counts no draft rounds, so no step length or `spec_drafts`); window prefill and decode tok/s from the totals' deltas, prefill = Δ(prompt − reused) ÷ Δ`prompt_ms`, decode = Δoutput ÷ Δ`decode_ms`, a total going down being a restart, into the existing engine speeds (tty11 backend line, exporter); the newest finished request's expert cache `hit_rate` and `pcie_share`. tty11's engine line gets a second line while Strata works: its `live.phase` (printable ASCII, at most 40 characters, never exported), then `prompt 4,096/10,240 (40 %)` while reading or `gen 14,558/32,000 · 29.6/s` while generating. RECENT keeps llama-swap's own speeds for Strata rows. The 1 MiB cap stays and `history` is never read. Snapshot (schema 1, additive): optional `engine.expert_hit` / `engine.pcie_share`, 0..=1; `engine.spec_drafts` is left out for an engine without rounds. llama-metrics: new gauges `llamabored_model_expert_cache_hit_ratio` and `llamabored_model_pcie_share_ratio`; Strata also fills the spec token counters, acceptance and the prefill/decode speed gauges. Goldens: exporter `loaded.prom` (a Strata model added to `snapshot-loaded.json`)
- SETUP for Strata (#54): a new rule source `engine:KEY` reads a setting the engine reports about itself, from a fixed list of keys the parser fills with numbers and short tokens (README "SETUP rules"); `name` reads a GGUF quant tag from the llama-swap model `name` (kind `quant`); kind `mib` draws MiB as `16.4 GiB`; live sources `expert_hit` and `pcie_share`. Built-in Strata rows: `engine Strata 0.1.41 · Q4_K_M`, `ctx 262,144 · kv q8 · resident 24,576`, `experts cache 14.6 GiB · 7,200 slots · hit 87 % · pcie 9 %`, `spec depth 5 · mtp 3 · lookup 2 · min-p 0.40 · 70 %`, `serve pcie 0.60 · arena 40.5 GiB · 12 workers · conv cache off`. Strata's ctx is now `engine.context` (else `max_context`). Golden: tty `setup-strata-160x49`

- tty11 SETUP block (#52): under the meters, the settings of the model generating now (else the one RECENT saw last, else the first loaded). Title `SETUP  <llama-swap id> · <name>` (the name only when it fits) and `+N` for the other loaded models; then one labelled row per setting (`engine`, `ctx`, `experts`, `spec`, `think`, `serve`, `sample`), items in order of importance. Six rows with the title up to 59 rows, eight from 60; rows drop from the bottom and items from the end of a row when short of room; under 48 rows only the rows beside SLOTS, so the short-screen panel order is unchanged; collapsed with nothing loaded or llama-swap down. Values nothing set are grey (`kv f16`, `full GPU`, `spec none`). Live speculative acceptance (`acc 3.6/step · 65 %`) and vLLM's `cache_config_info` facts come from the engine
- `[setup]` in watch.toml (#52): the rules that pick SETUP's values live in configuration. Each `[[setup.field]]` names a row, engines, an optional `match` on the launch command, a source (`flag:` names, `env:` name from `-e`/`--env`/`NAME=V`, `json:` flag and key one level deep, or `live:` engine/ctx/kv_dtype/kv_block/prefix_cache/spec_accept/spec_len), a kind (`number`, `token`, `quant`, `present`), and optional label, suffix, separator, map, default, fallback/group and order. Built-in rules (`crates/llama-watch/src/setup_defaults.toml`) cover upstream llama.cpp, vLLM and SGLang flags; watch.toml rules extend them and `defaults = false` drops them. Code keeps only numbers, short allowlisted tokens (file stem of a path-like value), quant tags and `on` from a command, whatever a rule says; the command is still read inside the `/running` decode and never kept. A bad rule fails validation with its index (`setup.field[3]: unknown kind "nubmer"`). `packaging/watch.example.toml` has a commented copy-paste block for an env-configured vLLM container (`SPEC`, `CTX`, `DFLASH_TOKENS`, `PREFIX_CACHE`, `MODEL`) and SGLang EXL3's `SGLANG_EXL3_MOE_OFFLOAD`. No new llama-swap path, no wire change

### Fixed

- A Strata server started as a container (`podman run … <image digest> --config …`) is now recognised (#54). Its launch command names no engine and the #31 `/metrics` probe knew only Prometheus prefixes, so it fell back to `openai` (exporter `backend="openai"`, no ctx, KV or live state). The probe now also tells Strata by the shape of its JSON `/metrics` (an object whose `engine` and `live` members are objects), once per load as before; `[llama.backends]` still overrides it. A detected Strata gets `running …/1` like a named one

### Changed

- tty11 meters are one per row (#52): CPU, GPU, VRAM, MEM, POWER, LOAD and ACTIVITY on rows 3-9 instead of every other row. Their bars are whole cells of `▇` (lower seven eighths, 3 pixels short of the cell in both llama-hack fonts) with `chart_glyphs = "eighths"`, and `▄` with `"halves"` (eurlatgr has no `▇`), so stacked bars stay apart; colours unchanged. 4K (90+ rows) keeps its two-row bars. llama-view and llama-cast already drew both glyphs; llama-cast gains a test that the shipped fonts leave the gap
- tty11's header detail is the engine alone (`model Qwen 35B  llama.cpp`); ctx, KV, quant, MoE, block and prefix moved to SETUP, so the header no longer runs into the clock (#52, #33's engine kept). `llama_core::detail::engine_items` is gone (no user left). Goldens: every tty frame (meters, SETUP, header), new `setup-160x49` and `setup-320x90`

## 0.3.5 — 2026-10-04

### Fixed

- tty11 SLOTS ctx sparklines are no longer wiped on every model swap (#45). A history is kept per (model display name, slot) and dropped only when its slot has not been on `/slots` for 30 min (or `tty.ctx_history_h`, if shorter), or to make room at the 64-slot cap (the slot away longest goes); a second model loading (A→A+B) leaves A's line alone, and A→B→A within that time resumes A's line, the time away shown as a gap. Only slots on `/slots` are drawn. A slot that comes back after the set of `/running` models changed gets the existing reset marker and drops its held context (the reloaded model's cache is gone); new lines after a swap are marked as before. The conversations a model's slots lost (#9's `evicted`) now outlive an unload and a llama-swap restart for their own 1 h, 16 per model, at most 64 models (oldest loss dropped first), so an eviction can be told across a swap. Drop counts behave as before. No golden changed
- tty11 RECENT no longer empties when llama-swap restarts (#44). llama-watch keeps its own ring of the newest 32 rows: each `/api/metrics/activity` read is merged in (new rows on top, rows already held updated in place) instead of replacing what is shown, so RECENT and its engine-measured speeds (#35) survive a restart and a model swap; RECENT still shows 8 rows with the text panels, up to 32 without. A llama-swap restart is now told by a lower top id, an empty list after rows, a held id whose fingerprint (model, timestamp, input and output tokens, duration) changed, or `/running` having been down with no held row on the page; its rows are a new generation and all count (previously a restart whose new top id reached the old one was skipped for fallback tokens, prompt-cache counters and speeds, and a reused id could take an old row's speeds). Old rows stay with their speeds; per-row state (speeds, reset-reason matching) is keyed by llama-watch's own row number, and a capture (IN/OUT) is fetched only for a row of the page just read, keyed by generation and id, so an old row's id is never fetched. Logged once per restart: `activity: llama-swap restarted; its row ids start again`
- llama-watch: engine and counter state no longer carries over or freezes across llama-swap restarts and engine changes (#46). A failed `/running` read now forgets the previously ready models' engine windows and attributed-speed reads at once (TTFT, ITL, spec and speeds from the old process stopped showing until a new window); the first good read after it still compares against the last good ready list. A model that leaves `ready`, or whose server changes under the same llama-swap id, counts its next process's decode and prompt counters from 0 (a reload already past the old value was undercounted by it) and goes back to activity rows for its prompt and cached counters until `/metrics` gives both again (a vLLM/SGLang → llama.cpp switch froze them). When `/running` is refused (llama-swap gone with its servers) the counters restart at once; a timeout keeps their baselines, so a busy llama-swap does not count a live server's totals twice
- llama-watch is no longer killed by its watchdog when a write to tty11 blocks (#42). The likely trigger on Titan was waking the blanked console with a key (Space, no XOFF): `con_write` waits on the console lock while fbcon and nvidia-drm unblank and modeset the monitor, 1 s to over 10 s, and the frame was written on the thread that publishes the snapshot and pings the watchdog (kraken-lcd and llama-light logged `snapshot stale` first). Frames now go to a `tty-writer` thread through a one-frame slot: the newest frame replaces a waiting one, publish and `WATCHDOG=1` never wait, a draw that took 1 s or more is followed by a full repaint, `tty: output stalled` / `tty: output resumed` are logged once each, and a clean stop waits at most 500 ms for a stuck writer. tty11 is also quietened at start: `ixon ixoff echo icanon isig` cleared and input flushed, only when stdout is `/dev/tty11` (Ctrl+S cannot pause it, keys do not echo onto it); a clean stop puts the saved flags back and `tty-reset` restores the console defaults before the palette. Scroll Lock still holds output (the dashboard freezes, then repaints); S13 allows `tcgetattr` / `tcsetattr` / `tcflush` in `term.rs`

## 0.3.4 — 2026-10-03

### Fixed

- tty11's IN no longer sticks on the first user message during an agent's tool loop (#38). For a llama-swap capture whose last message is not `user`, IN shows the `user` and `tool` messages after the last `assistant` one, in order, newest last, tool results marked `[tool]`, and the title says what they are: `IN (last request · 3 tool results)`, `IN (last request · user + 2 tool results)`. With no `assistant` message the run is every message but `system` / `developer`. A plain chat (last message `user`) and a completions prompt look as before. The walk still streams one message at a time and keeps at most 32,768 characters (the largest `llama.input_tail_chars`) of IN text; panel titles now draw the `·` separator. No golden changed
- tty11: a long model name and engine detail no longer run into the clock. The header detail drops whole trailing ` · item`s to keep two blank cells before the clock (at 160 columns `vLLM · kv kvarn_k4v2_g128 · block 128`, was `… · prefix2026-10-03 21:59:44`); the `slots` and `swap` fields are left out when they would reach it (#39)

## 0.3.3 — 2026-10-03

### Added

- The active inference engine is always named (#33). tty11's header detail
  starts with the shown model's engine (`llama.cpp · 256k · kv f16 · Q6_K`;
  `SGLang`, `vLLM`, `Strata`, `OpenAI-compatible`; a llama.cpp fork stays
  `llama.cpp`) and says `engine --` with no model. The LCD leads the detail
  line under a single model's name with the short name (`llama.cpp`,
  `sglang`, `vllm`, `strata`, `openai`), kept while quant, kv and
  `spec 78 %` drop for width. llama-view's pane title is
  `llama-view: <model> (<engine>) · <host>` when it fits the 40-character
  cap. The exporter is unchanged (`backend` label on
  `llamabored_model_loaded`).
- vLLM requests in RECENT get prompt and generation speeds measured by the
  engine (#35). llama-swap reports `-1` for them, so llama-watch reads vLLM's
  per-request histograms (`request_prefill_time_seconds`,
  `request_prefill_kv_computed_tokens`, `request_decode_time_seconds`,
  `request_generation_tokens`) on each metrics poll: prefill tok/s is
  uncached prompt tokens over prefill time, decode tok/s is generated tokens
  after the first over decode time. A new row gets the speeds of the
  requests that finished around it (their average if several), drawn
  `~2,134` / `~41.3` with a `~ = engine-measured` legend note; no match
  stays `--`, and llama.cpp rows are unchanged. SGLang has no per-request
  phase times and gets none. Snapshot: optional `engine.prefill_tps` /
  `engine.decode_tps` (schema 1, additive). llama-metrics:
  `llamabored_model_prefill_tokens_per_second` and
  `llamabored_model_decode_tokens_per_second`. tty11's backend line ends
  `· prefill 2,134/s · decode 41.2/s` where it fits.

## 0.3.2 — 2026-10-03

### Added

- A model whose launch command does not name its server (for example a
  container image that runs `vllm serve` itself) is recognised by its
  `/metrics` names, read once per load and only while llama-swap reports it
  `ready` (#31).
- vLLM: speculative-decoding acceptance rate and mean accepted length (tty11,
  the LCD and the exporter), mean TTFT, inter-token and request latency,
  preemptions, engine sleep, and the KV dtype, block size and prefix caching
  from `cache_config_info` for the tuning line. SGLang's spec-decode gauges
  and latency histograms are read too (#31).

## 0.3.1 — 2026-10-03

### Added

- tty11 loads llama-bored's own 16-colour palette, built from the heat ramp
  the LCD and RGB use (`[tty] palette = "llama"`, the default; `"vga"`
  keeps the kernel's colours). Only tty11 changes; `llama-watch tty-reset`
  restores the console palette when the unit stops. llama-cast renders with
  the same palette (`palette` in `cast.toml`) (#26).
- llama-view sends exact colours instead of the terminal theme's 16:
  `--colors auto|truecolor|256|16` (`LLAMA_VIEW_COLORS`) and
  `--palette llama|vga` (`LLAMA_VIEW_PALETTE`); `auto` uses truecolor when
  `COLORTERM` says so, else the nearest fixed xterm-256 colour (#25).
- llama-view is a better tmux resident: it follows the pane's and tty11's
  size (`--fit crop|center`, `--offset-x/-y`, a hint line when the pane is
  smaller), draws each frame as one synchronized update, drops to 1 fps
  while its pane is unfocused, sets the pane title to
  `llama-view: <model> · <host>` (and, opt-in, the tmux window name with
  `--tmux-window-name`), restores the terminal on `q`/Ctrl-C, and never
  rings the bell or captures the mouse (#27).

### Fixed

- llama-view decodes tty11's colours correctly with a 512-glyph console font
  (the bundled fonts); `--font-glyphs auto|256|512` overrides the detection (#28).
- The configured tty11 `[tty] font` and `size` now survive a reboot: at boot
  the framebuffer console's take-over can still be deferred, so
  `llama-watch tty-setup` writes one space to tty11 first and retries
  `setfont` for up to 5 s before setting the size (#22).

## 0.3.0 — 2026-09-30

### Added

- **llama-cast: watch tty11 on a TV.** A DLNA/UPnP media server that streams
  the tty11 dashboard as live 1920x1080 H.264 (MPEG-TS, rendered from the
  console at 2 fps with the bundled Hack font) to players on your LAN.
  Tested with Roku Media Player on TCL Roku TVs: open it, choose
  **llama-bored**, then **llama-bored live**. Off by default; its own user,
  an in-process CIDR allowlist plus `IPAddressAllow=`, pinned ports (a TCP
  stream port and UDP 1900 for SSDP), strict HTTP, one `ffmpeg` per viewer
  with a fixed argument list, no outbound connections, and a new source
  fence (S18). Setup: README "Watch tty11 on a TV".
- **Strata backend** ([Strata](https://github.com/Niko1221/Strata),
  `serve/server.py --engine strata`): detected from the launch command;
  tok/s, running and queued requests, prompt and cached-prompt counters and
  the context size from its JSON `/metrics`. Snapshot readers now read an
  unknown backend word as `openai` instead of rejecting the snapshot.

### Changed

- Only llama-metrics and llama-cast may listen (S16 now allows both, each
  under its own fence). The printed upgrade step restarts llama-cast too.

## 0.2.3 — 2026-09-30

### Fixed

- kraken-lcd: right after it opens the device, the Kraken can refuse bucket
  commands for several seconds (seen after a restart, while the previous
  process's stock screen settles). Until the first successful upload, for
  up to 60 s, refusals now back off without counting toward `fail_limit`;
  after that, failures count as before and persistent ones still restore
  the stock screen (#14).

## 0.2.2 — 2026-09-29

### Added

- Install without an NZXT Kraken Z: with none attached, the installer sets
  up everything except kraken-lcd's unit, udev rules and state directory,
  and says how to add it later; more than one still refuses (#6).
- tty11 on small or mixed screens: `[tty] size = "COLSxROWS"` and
  `[tty] font = "12x24" | "12x22"` (a second bundled Hack font), applied by
  a root pre-step `llama-watch tty-setup` with no shell; below 48 rows the
  layout drops IN/OUT, then FANS, then shrinks the chart instead of
  refusing, down to 160x26 (#7).
- SLOTS labels each context drop by best-guess reason (compacted, new,
  evicted, unknown) from token counts, with a legend (#9).
- llama-metrics: `llamabored_slot_ctx_used_tokens`,
  `llamabored_slot_ctx_resets_total{reason}`,
  `llamabored_model_prompt_tokens_total` and
  `llamabored_model_prompt_cached_tokens_total` (#10).
- IN/OUT for SGLang, vLLM and other servers without `/slots`: the last
  finished exchange from llama-swap's request captures
  (`/api/captures/<id>`, up to 2 MiB, never with `show_text = false`),
  pinned by a new source fence (S17) (#5).
- `scripts/demo/` and two ignored replay tests regenerate the README GIFs
  from recorded snapshots.

### Changed

- tty11 RECENT: DURATION sits beside the other numbers, left of the rate
  bar (#8).
- `llama-watch.service` loads the console font through `llama-watch
  tty-setup` instead of calling `setfont` directly.

### Fixed

- kraken-lcd stream mode backs off after a refused upload (2 s, doubling to
  30 s) instead of retrying on the next frame, so restarting it together
  with llama-watch no longer spends `fail_limit` in 300 ms and exits (#14).

## 0.2.1 — 2026-09-29

### Added

- llama-metrics exports everything the dashboards show: GPU and CPU power
  and the GPU power limit, VRAM and system memory in bytes, a prompt-token
  counter, llama.cpp slots busy and total, SGLang/vLLM running and queued
  requests, KV cache fill and cache hit ratio, model lifecycle state, fan
  rpm and pwm, and per-source health and latency. A test fails when a
  snapshot field is not exported (#11).

### Fixed

- Readers ignore unknown snapshot fields within the same schema number
  (known fields stay strictly validated), so upgrading llama-watch no
  longer blinds an older llama-metrics or llama-light; both now log once
  why they reject a snapshot. install.sh's printed restart step covers all
  four units (#12).

## 0.2.0 — 2026-09-29

### Added

- Backends beyond llama.cpp: SGLang, vLLM and any other OpenAI-compatible
  server behind llama-swap, told apart by the launch command or
  `[llama.backends]` in `watch.toml`. SGLang (start it with
  `--enable-metrics`) and vLLM metrics give tok/s, running and queued
  requests, KV fill and cache hit rate; other servers fall back to token
  counts from llama-swap's request log. The tuning line reads SGLang and
  vLLM flags. The snapshot gains optional `backend`, `running`, `queued`
  and `kv_fill` per model.
- tty11 shows the backend, a `running · queued · KV · hit` line in SLOTS
  for servers without `/slots`, and flags a model that stays `stopping`
  for over a minute.
- Continuous integration: GitHub Actions run `scripts/check.sh` on every
  push and pull request, a weekly advisory audit, and a release workflow
  that checks a `vX.Y.Z` tag against the crate versions and the changelog
  and publishes the release notes. Actions are pinned by commit; Dependabot
  keeps them current.

### Changed

- **Upgrading:** a llama.cpp server started through a wrapper script, with
  no `llama-server` in its llama-swap command, is now treated as a generic
  OpenAI-compatible server (no `/slots`, no llama.cpp metrics). Add
  `"model-id" = "llamacpp"` under `[llama.backends]` in `watch.toml`.
- **Upgrading:** `llamabored_model_loaded` has a new `backend` label
  (`llamacpp`, `sglang`, `vllm` or `openai`). Queries or alerts that match
  its exact label set need the label added or ignored.

## 0.1.2 — 2026-09-27

### Added

- `docs/RELEASING.md`: how versions, the changelog and releases work.
- README demo GIFs of the Kraken LCD, tty11 and the keyboard (`docs/media/`).
- README says up front what llama-bored needs: Linux with systemd and one
  NZXT Kraken Z (`1e71:3008`); NVIDIA, llama-swap, Aura and the keyboard are
  optional. Requirements list the tools the build and install scripts use
  (`cc`, `nm`, `ldd`, `sha256sum`, `setfacl`/`getfacl`).
- The `cpu_topk` light metric (the busiest cores) is documented in the
  README and the install skill.

### Changed

- Comments in the shipped units, udev rules and scripts no longer cite
  internal ticket numbers; risk ids name their row in `docs/SAFETY.md`.
- The install skill's frontmatter name is `install`, matching its folder,
  and it clones from `https://github.com/alexderz/llama-bored`.

### Fixed

- README, `docs/SAFETY.md` and `docs/ARCHITECTURE.md` describe device access
  exactly: kraken-lcd's usbfs grant is class-wide and narrowed by udev (new
  risk row RR7c), llama-light is pinned to two nodes (Aura and keyboard), and
  the unit hardening is listed per unit.
- The install skill's `pgrep -f` checks no longer match their own shell
  when run through `bash -c`.
- README and the skill name the log lines to look for: `cooling guard
  halted` / `cooling guard latch is present` (`journalctl -u kraken-lcd -p
  crit`) and llama-light's `aura absent (...)`. `check` commands are shown
  with their full path and `--config`.
- README: the demo generates at about 160 tok/s, not 80; tty11 is
  Ctrl+Alt+F11 from a graphical session; `listen` and `allow` in
  `metrics.toml` are required; the full uninstall also removes the
  `llama-metrics.service.d` drop-in directory.
- `docs/SAFETY.md`: the HALTED latch file is owned by kraken-lcd and
  cleared by an explicit root command; the development-only
  `--query-buckets` and `bench-upload` commands are guarded too.

### Security

- `llama-metrics.service` adds `InaccessiblePaths=/sys`, so the exporter
  cannot read sysfs even through a bug; it never needed it.

## 0.1.1 — 2026-09-27

### Added

- **llama-light drives the Corsair STRAFE RGB MK.2** (`1b1c:1b48`) per key,
  on its vendor lighting interface only (typing is untouched). A closed
  encoder with four report shapes (software mode, hardware mode, colour
  stream, 24-bit commit) and no profile, firmware or flash writes; stopping
  hands the keyboard back to its own lighting. An unplugged keyboard is
  logged once and looked for every 10 s. udev pins it to
  `/dev/llama-light/keyboard`, the unit's second `DeviceAllow=`.
- Keyboard targets: key ranges (`keyboard.keys["F1".."F12"]`), key lists,
  `keyboard.all`, single keys by name, and `led:N`.
- An optional frame pipeline for llama-light: `[engine]` (`tick_hz`,
  `target_hz`, `tween_fps`, `tween_s`) recomputes targets at a slow rate and
  tweens between them; `[base]` colour under every entry; named
  `[palette.NAME]` stops.
- New `[[light]]` options: `ladder`, `gate` and `peak` styles,
  `edge = "fractional"`, `gradient = "position"`, `attack_s` (asymmetric
  smoothing), `rate_window_s` (windowed tokens/s) and `brightness = [lo, hi]`.
- `packaging/light.example.toml` has a complete keyboard layout
  (`keyboard-c`), made for translucent keycaps where light bleeds between
  keys: activity on F1–F12, GPU and busiest cores as two bars each two rows
  tall (keys interleaved along the row stagger, so they grow in half-key
  steps), each bar one colour chosen by its value, GPU temperature on the
  navigation keys, and a tokens/s dial on the number pad around "5".

### Changed

- Colours: a deeper, more saturated cold end, and the top of the redline
  is a bright hot orange-gold instead of white. LCD, tty11 and RGB share it.
- tty11 level meters are spectra: each cell is coloured by its position.

## 0.1.0 — 2026-09-26

First public release. For watching when you're bored waiting on your agent.

Built with Claude Code and Grok: Claude (Opus, Fable) for design, review,
security and much of the code; Grok CLI for many of the builds.

### Added

- **llama-watch**, the single reader of `/proc`, hwmon, NVML and llama-swap,
  publishing a validated snapshot to `/run/llama-watch/snapshot.json` at
  10 Hz, and the **tty11 dashboard**:
  - the bundled Hack 12x24 console font and an eighth-block token chart with
    configurable rate ceilings (`gen_ceiling_tps`, `prompt_ceiling_tps`);
  - an ACTIVITY row that names its source (`gpu`, `cpu` or `util`);
  - SLOTS with per-slot context fill and a context-history sparkline with
    reset markers;
  - RECENT across the full width with full timestamps;
  - IN and OUT for the busy slot, with a clean prompt view that strips
    chat-template tokens;
  - `show_text = false` / `--no-text` to hide and stop keeping all llama text;
  - an optional read-only FANS panel for a Super-I/O hwmon chosen by name;
  - the full model name and a tuning line (ctx, KV cache, quant, CPU MoE).
- **Activity** as the bottleneck device's power headroom: GPU via NVML, CPU
  via zenergy, with `nominal_frac` and auto-learned idle floors that never
  rise; falls back to utilisation without power sensors.
- **llama-view**, a read-only mirror of tty11 for tmux or SSH (group
  `llama-view`).
- **kraken-lcd**, a network-free LCD writer for the NZXT Kraken Z53
  (320x320) with the "V3b Plasma Blackbody" face: a 270° activity gauge with
  a 100–125 redline, a 30-minute activity dial, a tokens·24h chart, model
  name and tuning line, temperatures, and distinct "AI down", "no model" and
  "no data" states. Change mode (default) and opt-in stream mode (10 fps).
  Its own minimal Kraken Z3 LCD protocol with a closed command set, a cooling
  guard, a persistent HALTED latch, stock restore on stop, and a guarded
  `show-image` test card.
- **llama-light**, a colour-only RGB writer for the ASUS Aura USB controller
  (`0b05:18f3`): `[[light]]` mappings from any snapshot metric to solid,
  ring, bar or pulse styles with palettes or colour stops, mirrored or
  daisy-chained fans, a brightness cap, and live reload of `light.toml`. A
  two-opcode encoder with no save-to-flash. Corsair keyboard support was
  detection only in this release.
- **llama-metrics**, a Prometheus exporter on TCP 19477 (`llamabored_*`
  series) for a CIDR allowlist, with strict HTTP limits; the only process
  that listens.
- **llama-core**, the shared snapshot schema v1, validator and sanitiser.
- Hardened systemd units, udev rules pinning each writer to one device node
  (`/dev/kraken-lcd/hid`, `/dev/llama-light/aura`), sysusers, and an
  installer anchored to `main` that verifies the build, never starts the LCD
  writer or enables the exporter, prints the next steps, and keeps a per-run
  rollback.
- Safety fences S1–S3 and S6–S16 in `scripts/check.sh` and the test suite.
- An agent install skill (`skills/install/SKILL.md`).

### Hardware

- Tested on the Kraken Z53 and an ASUS Aura USB controller. The Kraken Z63
  and Z73 share USB id `1e71:3008`, so they are accepted, but untested.
