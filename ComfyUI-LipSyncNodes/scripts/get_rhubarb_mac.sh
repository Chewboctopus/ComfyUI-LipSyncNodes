#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"/..
VER="1.14.0"
URL="https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v${VER}/rhubarb-${VER}-macOS.zip"
echo "Downloading Rhubarb ${VER}..."
curl -L "$URL" -o rhubarb-mac.zip
unzip -o rhubarb-mac.zip
rm rhubarb-mac.zip
echo "Rhubarb downloaded. Point 'rhubarb_path' to the extracted 'rhubarb' binary."
