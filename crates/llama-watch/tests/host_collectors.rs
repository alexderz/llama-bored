//! Host collectors on a synthetic proc and sys root.
//!
//! Nothing here opens the real `/proc` or `/sys`.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use llama_core::log::Sink;
use llama_core::sample::{AiState, LlamaView};
use llama_watch::collector::{WatchCollector, WatchSample};
use llama_watch::config::{Config, ValidWatchConfig};
use llama_watch::sources::gpu::{GpuBackend, GpuError};
use llama_watch::sources::hostio::{self, LinkSpeeds, Probe, SourceHealth};
use llama_watch::sources::{Roots, SourceId};

fn scratch(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "host-collectors-{label}-{}-{n}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("scratch");
    path
}

struct Tree {
    root: PathBuf,
}

impl Tree {
    fn new(label: &str) -> Self {
        let root = scratch(label);
        fs::create_dir_all(root.join("proc/pressure")).expect("proc");
        fs::create_dir_all(root.join("sys/class/net")).expect("sys");
        Self { root }
    }

    fn roots(&self) -> Roots {
        Roots {
            proc: self.root.join("proc"),
            sys: self.root.join("sys"),
        }
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent");
        }
        fs::write(path, text).expect(rel);
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn diskstats(read_sectors: u64, write_sectors: u64) -> String {
    // Field 6 is sectors read, field 10 is sectors written. The nvme row
    // has the 20 fields of a current kernel. The partition, loop and zram
    // rows are noise the sum must ignore.
    format!(
        " 259 0 nvme0n1 1 0 {read_sectors} 0 1 0 {write_sectors} 0 0 0 0 0 0 0 0 0 0\n\
         259 1 nvme0n1p1 9 0 999999 0 9 0 999999 0 0 0 0 0 0 0 0\n\
           8 0 sda 1 0 10 0 1 0 20 0 0 0 0 0 0 0 0\n\
           7 0 loop0 1 0 888888 0 1 0 888888 0 0 0 0 0 0 0 0\n\
         252 0 zram0 1 0 777777 0 1 0 777777 0 0 0 0 0 0 0 0\n\
         253 0 dm-0 1 0 666666 0 1 0 666666 0 0 0 0 0 0 0 0\n"
    )
}

/// `{sys}/class/net/<name>` with `speed`, `operstate` and, for a physical
/// interface, a `device` link.
fn iface(root: &Path, name: &str, speed: &str, operstate: &str, device: bool) {
    let dir = root.join("sys/class/net").join(name);
    fs::create_dir_all(&dir).expect("iface");
    fs::write(dir.join("speed"), speed).expect("speed");
    fs::write(dir.join("operstate"), operstate).expect("operstate");
    // Rewriting an interface keeps its link.
    if device && fs::symlink_metadata(dir.join("device")).is_err() {
        let target = root.join("sys/devices").join(name);
        fs::create_dir_all(&target).expect("device target");
        symlink(&target, dir.join("device")).expect("device link");
    }
}

const NET: &str = "\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 999999999 0 0 0 0 0 0 0 999999999 0 0 0 0 0 0 0
  eth0: 1000 1 0 0 0 0 0 0 2000 1 0 0 0 0 0 0
 wlan0: 50 1 0 0 0 0 0 0 70 1 0 0 0 0 0 0
docker0: 400 1 0 0 0 0 0 0 400 1 0 0 0 0 0 0
";

fn net_later() -> String {
    NET.replace("eth0: 1000", "eth0: 118001000")
        .replace("2000 1 0 0 0 0 0 0", "1002000 1 0 0 0 0 0 0")
}

/// `/proc/net/dev` with one physical `eth0` at `rx` and `tx` bytes.
fn eth0_dev(rx: u64, tx: u64) -> String {
    format!("  eth0: {rx} 1 0 0 0 0 0 0 {tx} 1 0 0 0 0 0 0\n")
}

fn read_net(tree: &Tree) -> Probe<hostio::NetCounters> {
    hostio::read_net(&tree.roots(), &mut LinkSpeeds::default(), Instant::now())
}

#[test]
fn diskstats_sums_whole_devices_and_ignores_partitions_loop_and_zram() {
    let tree = Tree::new("disk-sum");
    tree.write("proc/diskstats", &diskstats(1_000, 500));
    let Probe::Ready(counters) = hostio::read_disk(&tree.roots()) else {
        panic!("diskstats should parse");
    };
    // nvme0n1 1000 + sda 10. The partition, loop, zram and dm rows are out.
    assert_eq!(counters.read_sectors, 1_010);
    assert_eq!(counters.write_sectors, 520);
}

#[test]
fn a_short_line_the_sum_ignores_does_not_fail_diskstats() {
    let tree = Tree::new("disk-short-loop");
    tree.write(
        "proc/diskstats",
        &format!("{}   7 1 loop1 1 0\n", diskstats(1_000, 500)),
    );
    assert!(matches!(hostio::read_disk(&tree.roots()), Probe::Ready(_)));
}

#[test]
fn disk_rate_is_sector_delta_times_512() {
    let prev = 1_000u64;
    let next = prev + 23_437;
    let rate = hostio::sector_rate(prev, next, Duration::from_secs(1)).expect("rate");
    assert_eq!(rate, 23_437.0 * 512.0);
}

#[test]
fn a_disk_counter_that_goes_backwards_has_no_rate() {
    assert_eq!(
        hostio::sector_rate(500, 100, Duration::from_millis(100)),
        None
    );
    // A 5 s window plus a 5 s tick is the longest real span.
    assert_eq!(hostio::sector_rate(1, 2, Duration::from_secs(11)), None);
    assert_eq!(hostio::counter_rate(1, 2, Duration::from_secs(11)), None);
    assert_eq!(hostio::counter_rate(1, 2, Duration::ZERO), None);
}

#[test]
fn a_net_counter_that_goes_backwards_has_no_rate() {
    assert_eq!(
        hostio::counter_rate(500, 100, Duration::from_millis(100)),
        None
    );
}

#[test]
fn net_dev_counts_only_interfaces_with_a_device_link() {
    let tree = Tree::new("net-sum");
    tree.write("proc/net/dev", NET);
    iface(&tree.root, "eth0", "1000\n", "up\n", true);
    iface(&tree.root, "wlan0", "-1\n", "down\n", true);
    iface(&tree.root, "lo", "0\n", "unknown\n", false);
    iface(&tree.root, "docker0", "10000\n", "up\n", false);
    let Probe::Ready(net) = read_net(&tree) else {
        panic!("net/dev should parse");
    };
    // lo and docker0 have no device link. eth0 1000+2000 and wlan0 50+70.
    assert_eq!(net.rx_bytes, 1_050);
    assert_eq!(net.tx_bytes, 2_070);
    // eth0 is 1 Gbit. wlan0 is down and adds nothing.
    assert_eq!(net.ceiling_bps, Some(125_000_000));
}

#[test]
fn net_ceiling_sums_the_up_links_only() {
    let tree = Tree::new("net-up");
    tree.write(
        "proc/net/dev",
        "  enp5s0: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n\
           enp7s0: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n\
           wlp6s0: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n\
           usb0: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n\
           eno1: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n",
    );
    // Titan: a 2.5 G link up, a wired port with no cable, Wi-Fi off. usb0
    // is up with no speed and counts 1 Gbit. eno1 has no readable
    // operstate, which reads as unknown, so its 100 Mbit counts.
    iface(&tree.root, "enp5s0", "2500\n", "up\n", true);
    iface(&tree.root, "enp7s0", "-1\n", "down\n", true);
    iface(&tree.root, "wlp6s0", "", "down\n", true);
    iface(&tree.root, "usb0", "-1\n", "unknown\n", true);
    iface(&tree.root, "eno1", "100\n", "up\n", true);
    fs::remove_file(tree.root.join("sys/class/net/eno1/operstate")).expect("operstate");
    let Probe::Ready(net) = read_net(&tree) else {
        panic!("net");
    };
    assert_eq!(
        net.ceiling_bps,
        Some(312_500_000 + 125_000_000 + 12_500_000)
    );
}

#[test]
fn net_ceiling_is_link_speed_or_one_gbit() {
    let tree = Tree::new("net-speed");
    tree.write("proc/net/dev", &eth0_dev(1, 1));
    iface(&tree.root, "eth0", "not-a-number\n", "up\n", true);
    let Probe::Ready(net) = read_net(&tree) else {
        panic!("net");
    };
    assert_eq!(net.ceiling_bps, Some(hostio::ONE_GBIT_BPS as u64));

    // No link up at all: 1 Gbit, not zero.
    iface(&tree.root, "eth0", "10000\n", "down\n", true);
    let Probe::Ready(net) = read_net(&tree) else {
        panic!("net");
    };
    assert_eq!(net.ceiling_bps, Some(hostio::ONE_GBIT_BPS as u64));
}

#[test]
fn link_speeds_are_read_again_after_five_seconds_or_a_new_interface() {
    let tree = Tree::new("net-cache");
    tree.write("proc/net/dev", &eth0_dev(1, 1));
    iface(&tree.root, "eth0", "1000\n", "up\n", true);
    let mut links = LinkSpeeds::default();
    let t0 = Instant::now();
    let ceiling =
        |links: &mut LinkSpeeds, at: Instant| match hostio::read_net(&tree.roots(), links, at) {
            Probe::Ready(net) => net.ceiling_bps,
            other => panic!("{other:?}"),
        };
    assert_eq!(ceiling(&mut links, t0), Some(125_000_000));
    iface(&tree.root, "eth0", "10000\n", "up\n", true);
    assert_eq!(
        ceiling(&mut links, t0 + Duration::from_secs(1)),
        Some(125_000_000),
        "speed is not read every tick"
    );
    assert_eq!(
        ceiling(&mut links, t0 + Duration::from_secs(5)),
        Some(1_250_000_000)
    );
    tree.write(
        "proc/net/dev",
        &format!(
            "{}  eth1: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n",
            eth0_dev(1, 1)
        ),
    );
    iface(&tree.root, "eth1", "1000\n", "up\n", true);
    assert_eq!(
        ceiling(&mut links, t0 + Duration::from_secs(6)),
        Some(1_375_000_000),
        "a new interface is read at once"
    );
}

#[test]
fn an_interface_name_outside_ascii_parses() {
    let tree = Tree::new("net-utf8");
    tree.write(
        "proc/net/dev",
        "  eth0: 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n  wæn0: 5 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0\n",
    );
    iface(&tree.root, "eth0", "1000\n", "up\n", true);
    iface(&tree.root, "wæn0", "1000\n", "up\n", true);
    let Probe::Ready(net) = read_net(&tree) else {
        panic!("net");
    };
    assert_eq!((net.rx_bytes, net.tx_bytes), (6, 8));
}

/// #163 M7: the file is there but nothing in it qualifies. That is
/// absent, not a zero rate.
#[test]
fn nothing_to_count_is_absent_not_zero() {
    let tree = Tree::new("nothing-to-count");
    tree.write(
        "proc/diskstats",
        " 259 1 nvme0n1p1 9 0 999999 0 9 0 999999 0 0 0 0 0 0 0 0\n\
           7 0 loop0 1 0 888888 0 1 0 888888 0 0 0 0 0 0 0 0\n",
    );
    tree.write("proc/net/dev", NET);
    iface(&tree.root, "lo", "0\n", "unknown\n", false);
    iface(&tree.root, "docker0", "10000\n", "up\n", false);
    assert!(matches!(hostio::read_disk(&tree.roots()), Probe::Missing));
    assert!(matches!(read_net(&tree), Probe::Missing));
}

#[test]
fn net_rate_is_the_byte_delta() {
    let rate = hostio::counter_rate(1_000, 118_001_000, Duration::from_secs(1)).expect("rx");
    assert_eq!(rate, 118_000_000.0);
}

#[test]
fn swap_used_is_total_minus_free_in_bytes() {
    let tree = Tree::new("swap");
    tree.write(
        "proc/meminfo",
        "MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 16384 kB\nSwapFree: 12288 kB\n",
    );
    let Probe::Ready(swap) = hostio::read_swap(&tree.roots()) else {
        panic!("swap");
    };
    assert_eq!(swap.total, 16_384 * 1024);
    assert_eq!(swap.used, 4_096 * 1024);
}

#[test]
fn meminfo_without_swap_keys_is_absent() {
    let tree = Tree::new("no-swap");
    tree.write("proc/meminfo", "MemTotal: 1024 kB\nMemAvailable: 512 kB\n");
    assert!(matches!(hostio::read_swap(&tree.roots()), Probe::Missing));
}

#[test]
fn psi_reads_some_avg10_from_cpu_io_and_memory() {
    let tree = Tree::new("psi");
    tree.write(
        "proc/pressure/cpu",
        "some avg10=2.50 avg60=1.00 avg300=0.10 total=5\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
    );
    tree.write(
        "proc/pressure/io",
        "some avg10=1.10 avg60=0.40 avg300=0.10 total=9\nfull avg10=0.20 avg60=0.00 avg300=0.00 total=1\n",
    );
    tree.write(
        "proc/pressure/memory",
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
    );
    let Probe::Ready(psi) = hostio::read_psi(&tree.roots()) else {
        panic!("psi");
    };
    assert_eq!(psi.cpu, Some(2.5));
    assert_eq!(psi.io, Some(1.1));
    assert_eq!(psi.memory, Some(0.0));
}

#[test]
fn missing_pressure_files_are_absent() {
    let tree = Tree::new("no-psi");
    fs::remove_dir_all(tree.root.join("proc/pressure")).expect("drop pressure");
    assert!(matches!(hostio::read_psi(&tree.roots()), Probe::Missing));
    assert!(matches!(hostio::read_disk(&tree.roots()), Probe::Missing));
    assert!(matches!(read_net(&tree), Probe::Missing));
}

#[test]
fn a_malformed_diskstats_line_fails() {
    let tree = Tree::new("bad-disk");
    tree.write("proc/diskstats", "259 0 nvme0n1 short\n");
    assert!(matches!(hostio::read_disk(&tree.roots()), Probe::Failed(_)));
}

#[derive(Clone, Copy)]
struct QuietGpu;

impl GpuBackend for QuietGpu {
    fn init(&mut self) -> Result<(), GpuError> {
        Ok(())
    }
    fn util_pct(&mut self) -> Result<f32, GpuError> {
        Ok(0.0)
    }
    fn temp_c(&mut self) -> Result<f32, GpuError> {
        Ok(40.0)
    }
    fn disconnect(&mut self) {}
}

struct Discard;

impl Sink for Discard {
    fn write_line(&mut self, _line: &str) {}
}

fn watch_config(dir: &Path) -> ValidWatchConfig {
    watch_config_window(dir, 1.0)
}

fn watch_config_window(dir: &Path, cpu_window_s: f64) -> ValidWatchConfig {
    let path = dir.join("watch.toml");
    fs::write(
        &path,
        format!(
            "[collector]\ntick_s = 0.1\ncpu_window_s = {cpu_window_s:?}\ncpu_top_k = 1\n\
         [llama]\nurl = \"http://127.0.0.1:9\"\n\
         [load]\nsmooth_s = 0\n\
         [temps]\nenabled = false\n\
         [fans]\nenabled = false\n"
        ),
    )
    .expect("config");
    Config::load_validated(&path, 1).expect("valid")
}

fn idle_view() -> LlamaView {
    LlamaView {
        ai: AiState::Idle,
        models: Vec::new(),
        decoded_total: None,
        prompt_total: None,
    }
}

/// A host with proc/stat and meminfo, so the collector has CPU and memory.
fn host_tree(label: &str) -> Tree {
    let tree = Tree::new(label);
    tree.write(
        "proc/stat",
        "cpu  0 0 0 0 0 0 0 0\ncpu0 0 0 0 100 0 0 0 0\n",
    );
    tree.write(
        "proc/meminfo",
        "MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 16384 kB\nSwapFree: 12288 kB\n",
    );
    tree
}

/// One collector tick `after` the start, with the files as they are now.
fn tick(
    collector: &mut WatchCollector<'_, QuietGpu, Discard>,
    t0: Instant,
    after: Duration,
) -> WatchSample {
    collector.sample(t0 + after, SystemTime::UNIX_EPOCH, &idle_view())
}

#[test]
fn the_collector_publishes_disk_and_net_rates_once_a_sample_is_a_window_old() {
    let tree = host_tree("loop");
    tree.write("proc/diskstats", &diskstats(1_000, 500));
    tree.write("proc/net/dev", NET);
    iface(&tree.root, "eth0", "1000\n", "up\n", true);
    iface(&tree.root, "wlan0", "-1\n", "down\n", true);
    tree.write(
        "proc/pressure/io",
        "some avg10=1.10 avg60=0 avg300=0 total=1\n",
    );
    tree.write(
        "proc/pressure/memory",
        "some avg10=0.50 avg60=0 avg300=0 total=1\n",
    );
    tree.write(
        "proc/pressure/cpu",
        "some avg10=2.50 avg60=0 avg300=0 total=1\n",
    );
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    let first = tick(&mut collector, t0, Duration::ZERO);
    assert_eq!(first.host_io.disk, SourceHealth::Ok);
    assert_eq!(first.host_io.net, SourceHealth::Ok);
    assert_eq!(first.host_io.psi, SourceHealth::Ok);
    assert_eq!(first.host_io.disk_read_bps, None, "no rate before a delta");
    assert_eq!(first.host_io.net_rx_bps, None, "no rate before a delta");
    assert_eq!(first.host_io.swap_total_bytes, Some(16_384 * 1024));
    assert_eq!(first.host_io.swap_used_bytes, Some(4_096 * 1024));
    assert_eq!(first.host_io.io_psi, Some(1.1));
    assert_eq!(first.host_io.mem_psi, Some(0.5));
    assert_eq!(first.host_io.cpu_psi, Some(2.5));
    assert!(
        !first.snapshot.errors.contains(&SourceId::ProcDisk),
        "{:?}",
        first.snapshot.errors
    );

    tree.write("proc/diskstats", &diskstats(1_000 + 23_437, 500 + 1_000));
    tree.write("proc/net/dev", &net_later());
    let early = tick(&mut collector, t0, Duration::from_millis(500));
    assert_eq!(
        early.host_io.disk_read_bps, None,
        "no stored sample is a window (1 s) old yet"
    );
    let second = tick(&mut collector, t0, Duration::from_secs(1));
    assert_eq!(second.host_io.disk_read_bps, Some(23_437.0 * 512.0));
    assert_eq!(second.host_io.disk_write_bps, Some(1_000.0 * 512.0));
    // eth0 rx grew by 118_000_000. wlan0 did not move. lo is not counted.
    assert_eq!(second.host_io.net_rx_bps, Some(118_000_000.0));
    assert_eq!(second.host_io.net_tx_bps, Some(1_000_000.0));
    assert_eq!(second.host_io.net_ceiling_bps, Some(125_000_000.0));
}

/// #163 M6: writeback in bursts every other 0.1 s tick is 0 and 2.56 GB/s
/// tick by tick. Over the one-second window it is one steady number.
#[test]
fn bursty_disk_and_net_read_steady_over_the_cpu_window() {
    let tree = host_tree("steady");
    iface(&tree.root, "eth0", "10000\n", "up\n", true);
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    let burst = 500_000u64;
    let (mut sectors, mut bytes) = (0u64, 0u64);
    for k in 0..=30u64 {
        if k % 2 == 1 {
            sectors += burst;
            bytes += burst * 512;
        }
        tree.write("proc/diskstats", &diskstats(0, sectors));
        tree.write("proc/net/dev", &eth0_dev(bytes, 0));
        let sample = tick(&mut collector, t0, Duration::from_millis(100 * k));
        if k < 10 {
            assert_eq!(sample.host_io.disk_write_bps, None, "tick {k}");
            assert_eq!(sample.host_io.net_rx_bps, None, "tick {k}");
        } else {
            let steady = 5.0 * burst as f64 * 512.0;
            assert_eq!(sample.host_io.disk_write_bps, Some(steady), "tick {k}");
            assert_eq!(sample.host_io.net_rx_bps, Some(steady), "tick {k}");
        }
    }
}

/// `collector.cpu_window_s` may be 5 s. Ticks run a little late, so every
/// baseline is a little over 5 s old: that is still a rate, not a stall.
#[test]
fn a_five_second_window_still_rates_when_ticks_run_late() {
    let tree = host_tree("window-5");
    let config = watch_config_window(&tree.root, 5.0);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    let per_tick = 1_000u64;
    for k in 0..=80u64 {
        let ms = 103 * k;
        tree.write("proc/diskstats", &diskstats(per_tick * k, 0));
        let rate = tick(&mut collector, t0, Duration::from_millis(ms))
            .host_io
            .disk_read_bps;
        if ms < 5_000 {
            assert_eq!(rate, None, "tick {k}: the window is not full");
        } else {
            let want = per_tick as f64 * 512.0 / 0.103;
            let got = rate.unwrap_or_else(|| panic!("tick {k} at {ms} ms has no rate"));
            assert!(
                (got - want).abs() < want * 1e-9,
                "tick {k}: {got} vs {want}"
            );
        }
    }
}

/// A counter that drops (a device unplugged) starts a new baseline: no
/// rate spans the drop, even against an older, lower baseline.
#[test]
fn a_net_counter_drop_starts_a_new_baseline() {
    let tree = host_tree("net-drop");
    iface(&tree.root, "eth0", "1000\n", "up\n", true);
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    let mut rx_at = |rx: u64, ms: u64| {
        tree.write("proc/net/dev", &eth0_dev(rx, 0));
        tick(&mut collector, t0, Duration::from_millis(ms))
            .host_io
            .net_rx_bps
    };
    assert_eq!(rx_at(100, 0), None);
    assert_eq!(rx_at(1_500, 500), None);
    // 300 is above the 1 s old 100, but the counter fell from 1500.
    assert_eq!(rx_at(300, 1_000), None, "no rate across the drop");
    assert_eq!(rx_at(400, 1_500), None, "the new baseline is not 1 s old");
    assert_eq!(rx_at(1_300, 2_000), Some(1_000.0));
}

/// A whole-disk counter that drops (a disk pulled, read or write) starts a
/// new baseline, as a net counter does.
#[test]
fn a_disk_counter_drop_starts_a_new_baseline() {
    for write_drops in [false, true] {
        let tree = host_tree("disk-drop");
        let config = watch_config(&tree.root);
        let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
        let t0 = Instant::now();
        let mut read_at = |read: u64, write: u64, ms: u64| {
            tree.write("proc/diskstats", &diskstats(read, write));
            tick(&mut collector, t0, Duration::from_millis(ms))
                .host_io
                .disk_read_bps
        };
        // Only the counter named by the case falls at 1 s: read 1500 to
        // 300 with write flat, or write 1500 to 100 with read climbing.
        let (peak, fell_write) = if write_drops {
            (200, 100)
        } else {
            (1_500, 1_500)
        };
        assert_eq!(read_at(100, 1_000, 0), None);
        assert_eq!(read_at(peak, 1_500, 500), None);
        assert_eq!(
            read_at(300, fell_write, 1_000),
            None,
            "no rate across the drop (write drops: {write_drops})"
        );
        assert_eq!(read_at(400, fell_write, 1_500), None);
        assert_eq!(read_at(1_300, fell_write, 2_000), Some(1_000.0 * 512.0));
    }
}

/// /proc/net/dev gone for a tick: the next reading is a first sample, not
/// a rate against the counters from before the gap.
#[test]
fn net_dev_gone_for_a_tick_starts_a_new_baseline() {
    let tree = host_tree("net-gone");
    iface(&tree.root, "eth0", "1000\n", "up\n", true);
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    tree.write("proc/net/dev", &eth0_dev(100, 0));
    assert_eq!(
        tick(&mut collector, t0, Duration::ZERO).host_io.net_rx_bps,
        None
    );
    fs::remove_file(tree.root.join("proc/net/dev")).expect("net/dev");
    let gone = tick(&mut collector, t0, Duration::from_millis(500));
    assert_eq!(gone.host_io.net, SourceHealth::Absent);
    tree.write("proc/net/dev", &eth0_dev(1_100, 0));
    let back = tick(&mut collector, t0, Duration::from_secs(1));
    assert_eq!(back.host_io.net_rx_bps, None, "a first sample again");
    tree.write("proc/net/dev", &eth0_dev(2_100, 0));
    let next = tick(&mut collector, t0, Duration::from_secs(2));
    assert_eq!(next.host_io.net_rx_bps, Some(1_000.0));
}

#[test]
fn a_disk_gap_over_five_seconds_has_no_rate() {
    let tree = host_tree("disk-gap");
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    let mut read_at = |sectors: u64, ms: u64| {
        tree.write("proc/diskstats", &diskstats(sectors, 0));
        tick(&mut collector, t0, Duration::from_millis(ms))
            .host_io
            .disk_read_bps
    };
    assert_eq!(read_at(1_000, 0), None);
    assert_eq!(read_at(2_000, 6_000), None, "a 6 s stall is not a rate");
    assert_eq!(read_at(3_000, 7_000), Some(1_000.0 * 512.0));
}

/// The stall edge: a gap of exactly 5 s between two samples is still a
/// rate, a gap past it starts a new baseline.
#[test]
fn a_gap_of_five_seconds_is_a_rate_and_past_it_is_a_stall() {
    for (gap_ms, rated) in [(5_000u64, true), (5_001, false)] {
        let tree = host_tree("disk-edge");
        let config = watch_config(&tree.root);
        let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
        let t0 = Instant::now();
        tree.write("proc/diskstats", &diskstats(1_000, 0));
        tick(&mut collector, t0, Duration::ZERO);
        tree.write("proc/diskstats", &diskstats(6_000, 0));
        let rate = tick(&mut collector, t0, Duration::from_millis(gap_ms))
            .host_io
            .disk_read_bps;
        assert_eq!(rate.is_some(), rated, "{gap_ms} ms gap: {rate:?}");
    }
}

/// A stall with the longest window: no rate until a sample after the
/// stall is a whole 5 s window old, and none spans the stall.
#[test]
fn a_stall_with_a_five_second_window_waits_a_window() {
    let tree = host_tree("disk-stall-5");
    let config = watch_config_window(&tree.root, 5.0);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    let mut read_at = |sectors: u64, ms: u64| {
        tree.write("proc/diskstats", &diskstats(sectors, 0));
        tick(&mut collector, t0, Duration::from_millis(ms))
            .host_io
            .disk_read_bps
    };
    for k in 0..=50u64 {
        let rate = read_at(1_000 * k, 100 * k);
        assert_eq!(rate.is_some(), k == 50, "warm-up tick {k}");
    }
    // 5.1 s with no sample, and a burst in it that must not be rated.
    let stall = 5_000 + 5_100;
    assert_eq!(read_at(10_000_000, stall), None, "the stall tick");
    for k in 1..=50u64 {
        let rate = read_at(10_000_000 + 1_000 * k, stall + 100 * k);
        if k < 50 {
            assert_eq!(rate, None, "tick {k} after the stall");
        } else {
            assert_eq!(
                rate,
                Some(50_000.0 * 512.0 / 5.0),
                "a window after the stall"
            );
        }
    }
}

#[test]
fn a_failed_diskstats_read_restarts_the_rate() {
    let tree = host_tree("disk-fail");
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    tree.write("proc/diskstats", &diskstats(1_000, 0));
    assert_eq!(
        tick(&mut collector, t0, Duration::ZERO).host_io.disk,
        SourceHealth::Ok
    );
    tree.write("proc/diskstats", "259 0 nvme0n1 short\n");
    let down = tick(&mut collector, t0, Duration::from_secs(1));
    assert_eq!(down.host_io.disk, SourceHealth::Down);
    assert!(down.snapshot.errors.contains(&SourceId::ProcDisk));
    tree.write("proc/diskstats", &diskstats(5_000, 0));
    let back = tick(&mut collector, t0, Duration::from_secs(2));
    assert_eq!(back.host_io.disk, SourceHealth::Ok);
    assert_eq!(back.host_io.disk_read_bps, None, "a first sample again");
    tree.write("proc/diskstats", &diskstats(6_000, 0));
    let next = tick(&mut collector, t0, Duration::from_secs(3));
    assert_eq!(next.host_io.disk_read_bps, Some(1_000.0 * 512.0));
}

/// #163 M7 through the collector: no whole disk and no physical
/// interface leave the rates and the ceiling absent, and nothing in
/// `errors`.
#[test]
fn the_collector_leaves_nothing_to_count_absent() {
    let tree = host_tree("absent-loop");
    tree.write(
        "proc/diskstats",
        "   7 0 loop0 1 0 888888 0 1 0 888888 0 0 0 0 0 0 0 0\n",
    );
    tree.write("proc/net/dev", NET);
    let config = watch_config(&tree.root);
    let mut collector = WatchCollector::new(tree.roots(), QuietGpu, &config, Discard);
    let t0 = Instant::now();
    for secs in 0..3 {
        let sample = tick(&mut collector, t0, Duration::from_secs(secs));
        assert_eq!(sample.host_io.disk, SourceHealth::Absent);
        assert_eq!(sample.host_io.net, SourceHealth::Absent);
        assert_eq!(sample.host_io.disk_read_bps, None);
        assert_eq!(sample.host_io.disk_write_bps, None);
        assert_eq!(sample.host_io.net_rx_bps, None);
        assert_eq!(sample.host_io.net_tx_bps, None);
        assert_eq!(sample.host_io.net_ceiling_bps, None);
        let errors = &sample.snapshot.errors;
        assert!(
            !errors.contains(&SourceId::ProcDisk) && !errors.contains(&SourceId::ProcNet),
            "{errors:?}"
        );
    }
}
