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
  the snapshot bumps it.

## Changelog

- Every change that a user could notice adds a line under `## Unreleased`,
  in `Added`, `Changed`, `Fixed`, `Removed` or `Security`, in the same
  commit as the change.
- A release renames `## Unreleased` to `## X.Y.Z — YYYY-MM-DD` and opens a
  new, empty `## Unreleased` above it.

## Cutting a release

1. `main` is green: `scripts/check.sh` exits 0 on a clean tree.
2. Bump the version (above), move `Unreleased` to the new version, and
   update the README, `docs/` and `skills/install/SKILL.md` where the
   release changes what they describe.
3. Commit `llama-bored X.Y.Z: <one-line summary>` and run `scripts/check.sh`
   again on the committed tree.
4. Tag it annotated: `git tag -a vX.Y.Z -m "llama-bored X.Y.Z"`, then
   `git push origin main vX.Y.Z`.
5. Publish the GitHub release with that changelog section as its notes:
   `gh release create vX.Y.Z --title "llama-bored X.Y.Z" --notes-file <section>`.

Never move a pushed tag, rewrite pushed history or force-push `main`. A
mistake in a release is fixed by the next patch release.
