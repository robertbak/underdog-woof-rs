#!/usr/bin/env bash
# Speak text through the machine's default audio sink, streaming as it is synthesised.
#
#   say "Your order ships on March 3rd."          # bare text, or --text "..."
#   say --responsive "Longer text, heard quickly."  # flags and text together
#   echo "Piped text." | say                        # or a whole file: cat notes.txt | say
#   tools/say.sh --voice mira --text "…"            # from the repo, same script
#
# No weights needed: with nothing to point at, woof-rs fetches the reference .npz from upstream once
# and caches it. Point it elsewhere with WOOF_MODEL=/path/to/woof_int8.npz.
#
# `--responsive` (or WOOF_RESPONSIVE=1) turns on the low-latency preset: the first utterance is
# decoded every 13 latent frames (~0.5 s of audio) with a single take, which takes time-to-first-
# sound from ~0.9 s to ~0.25 s on a desktop (see `woof-rs --bench`). It costs ~2.8x throughput, so
# it is applied to the first utterance only.
#
# Extra flags are passed to woof-rs, and with no text argument the text is read from stdin, so
# `echo hello | say` and `cat notes.txt | say` both work. The audio path is
# woof-rs --pcm-stdout --stream | <player>, so the first audio is audible while the rest is still
# being generated. Note that this *plays*: piping say into `head` truncates the player's stdout, not
# the audio. To hear the first few seconds of something long, cap the producer instead:
#   timeout 3 say < notes.txt
#   woof-rs --pcm-stdout --stream < notes.txt | head -c 96000 > sample.pcm
# Killing the player stops synthesis within a few frames (woof-rs gets SIGPIPE).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=${WOOF_RS_BIN:-$(command -v woof-rs || echo "$here/target/release/woof-rs")}
# weights: WOOF_MODEL, else a copy beside the checkout, else let the binary fetch and cache them
model_args=()
if [ -n "${WOOF_MODEL:-}" ]; then
    model_args=(--model "$WOOF_MODEL")
elif [ -f "$here/models/woof_int8.npz" ]; then
    model_args=(--model "$here/models/woof_int8.npz")
elif [ -f "$here/models/woof-i8.json" ]; then
    model_args=(--model "$here/models/woof-i8")
fi

[ -x "$bin" ] || { echo "say.sh: build first (cargo build --release) or set WOOF_RS_BIN" >&2; exit 1; }

# 24 kHz mono 16-bit signed little-endian, which is what the model emits.
if command -v pw-cat >/dev/null; then
    # PipeWire: --raw keeps pw-cat from trying to sniff a file format on stdin
    player=(pw-cat --playback --raw --format=s16 --rate=24000 --channels=1 -)
elif command -v pacat >/dev/null; then
    # PulseAudio, with a small buffer so playback starts promptly
    player=(pacat --format=s16le --rate=24000 --channels=1 --latency-msec=120)
elif command -v aplay >/dev/null; then
    # ALSA: 1024-frame periods (~43 ms) keep latency low without underruns
    player=(aplay -f S16_LE -r 24000 -c 1 -q --period-size=1024 --buffer-size=4096)
else
    echo "say.sh: no player found (install pipewire-utils, pulseaudio-utils or alsa-utils)" >&2
    exit 1
fi

# strip --responsive (or WOOF_RESPONSIVE) before passing the rest through to woof-rs
args=()
for arg in "$@"; do
    if [ "$arg" = "--responsive" ]; then responsive=1; else args+=("$arg"); fi
done

preset=()
if [ -n "${WOOF_RESPONSIVE:-}" ] || [ -n "${responsive:-}" ]; then
    preset=(--chunk-frames 13 --no-retry-first)
fi

"$bin" "${model_args[@]}" "${preset[@]}" "${args[@]}" --pcm-stdout --stream 2> >(sed 's/^/woof: /' >&2) | "${player[@]}"
