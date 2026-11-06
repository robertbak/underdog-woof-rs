#!/usr/bin/env bash
# Build the release binary, install it to ~/bin, and drop the build tree.
#
#   tools/install.sh            # cargo build --release -> ~/bin/woof-rs -> cargo clean
#   INSTALL_DIR=/usr/local/bin tools/install.sh
#
# Keeping `target/` clean is why the repo ships a single 4.4 MiB binary instead of 1.5 GiB of
# intermediates: every consumer (llama-swap's seat, the Python fallback, the test tools) uses the
# installed copy, so a stale `target/` is only ever a build artefact.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dest=${INSTALL_DIR:-$HOME/bin}

cd "$here"
cargo build --release
install -Dm755 target/release/woof-rs "$dest/woof-rs"
echo "installed $dest/woof-rs ($(stat -c %s "$dest/woof-rs" | numfmt --to=iec-i --suffix=B))"

# speak text through the default sink; it finds the binary on PATH and the weights by absolute path,
# so it works from any directory (and survives `cargo clean`)
install -Dm755 tools/say.sh "$dest/say.sh"
ln -sf "$dest/say.sh" "$dest/say"
echo "installed $dest/say.sh (+ symlink $dest/say)"

if [ "${KEEP_TARGET:-0}" != "1" ]; then
    cargo clean
    echo "cleaned $here/target"
fi
