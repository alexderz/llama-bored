# Releasing

llama-bored follows [Semantic Versioning](https://semver.org/) and keeps a
[changelog](../CHANGELOG.md) in the style of
[Keep a Changelog](https://keepachangelog.com/).

## Versions

- While the version is 0.x: a **patch** (0.1.x) adds features and fixes
  without breaking anything a user set up; a **minor** (0.x.0) may change a
  config key, the snapshot schema, a metric name, a unit, a udev link or a
  path, and its changelog says what to change.
- All crates share one version. Bump every `crates/*/Cargo.toml` and
  refresh `Cargo.lock` (`cargo update --offline --workspace`).
- The snapshot has its own `schema` number. Only an incompatible change to
  the snapshot bumps it. Within one schema number, new fields are optional
  and readers ignore fields they do not know (known fields stay strictly
  validated, and the size cap stays), so a newer llama-watch never blinds
  an older kraken-lcd, llama-light or llama-metrics.
- After an upgrade, restart every running unit so all of them run the new
  binaries: `systemctl try-restart llama-watch kraken-lcd llama-light
  llama-metrics llama-cast` (install.sh prints this; it never restarts units itself).

## Changelog

- Every change that a user could notice adds a line under `## Unreleased`,
  in `Added`, `Changed`, `Fixed`, `Removed` or `Security`, in the same
  commit as the change.
- A release renames `## Unreleased` to `## X.Y.Z — YYYY-MM-DD` and opens a
  new, empty `## Unreleased` above it.

## Changes

`main` is protected: every change lands through a pull request, and the
`check.sh` CI job must pass first. CI runs `scripts/check.sh` on each push
and pull request; a weekly job runs `cargo audit` and `cargo deny check
advisories` so new advisories show up without a code change.

## Cutting a release

1. `main` is green: CI passes, and `scripts/check.sh` exits 0 on a clean
   tree locally.
2. Bump the version (above), move `Unreleased` to the new version, and
   update the README, `docs/` and `skills/install/SKILL.md` where the
   release changes what they describe.
3. Commit `llama-bored X.Y.Z: <one-line summary>` on a branch, open a pull
   request, and merge it when CI passes.
4. Tag the merged commit on `main`, annotated:
   `git tag -a vX.Y.Z -m "llama-bored X.Y.Z"`, then `git push origin vX.Y.Z`.
5. The release workflow checks that the tag matches every crate version
   and that `CHANGELOG.md` has a `## X.Y.Z` section, then publishes the
   GitHub release with that section as its notes.

Never move a pushed tag, rewrite pushed history or force-push `main`. A
mistake in a release is fixed by the next patch release.
