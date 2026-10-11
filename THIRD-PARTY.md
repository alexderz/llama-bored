# Third-party software

This file states what this repository ships. It is not legal advice.

## ffmpeg

llama-cast runs the `ffmpeg` program already on the machine (`/usr/bin/ffmpeg`
by default, or the absolute path set in `cast.toml`). The operator installs
that program from their distribution. A container that runs llama-cast pulls
ffmpeg from its own packager when the image is built or when the container
starts.

llama-bored does not bundle, link, vendor or redistribute ffmpeg or any
encoder. None of their source and none of their binaries is in this
repository. llama-cast starts `ffmpeg` as a separate program, one process
per viewer, with a fixed argument list and an empty environment.

ffmpeg is published under the LGPL or the GPL, depending on how that binary
was built. A build that includes libx264 is GPL. Codec patent terms, where
they apply, apply to that binary and to whoever distributes it. They do not
apply to llama-bored, which is MIT. Anyone who packages ffmpeg together with
llama-bored has to follow ffmpeg's terms for that distribution.

## Hack

The console fonts `packaging/fonts/llama-hack-12x24.psfu`,
`packaging/fonts/llama-hack-12x22.psfu` and
`packaging/fonts/llama-hack-10x18.psfu` are rendered from Hack Regular.
Their licence is [packaging/fonts/LICENSE.Hack](packaging/fonts/LICENSE.Hack).
