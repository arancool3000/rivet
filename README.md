# Rivet

A local AI chat app for Apple Silicon Macs. It runs models through LM Studio (MLX) and serves a chat UI on `http://localhost:4480`.

## Models

Four models, named after boards (bigger board = bigger model). The weights are in this repo's **Releases**, split into parts under 2 GB.

| Folder | Base model | Size | RAM when loaded |
|---|---|---|---|
| `1-Apex-Gemma4-26B` (Pi 5) | Gemma 4 26B A4B | 15.6 GB | ~16 GB |
| `4-Pi4-Gemma4-26B-3bit` (Pi 4) | Gemma 4 26B A4B, tuned 3-bit | 12.8 GB | ~14 GB |
| `2-Forge-gpt-oss-20b` (Pi 3) | gpt-oss 20B, tuned 3-bit experts | 10.3 GB | ~10.5 GB |
| `3-ESP32-Qwen3.5-4B` (ESP32-S3) | Qwen3.5 4B | 3.1 GB | ~3.5 GB |

On a 16 GB Mac use Pi 3 or ESP32-S3. Pi 4 and Pi 5 need 24 GB or more.

### Download and rebuild a model

```bash
brew install gh
gh release download models -R arancool3000/rivet -p '2-Forge-gpt-oss-20b*'
./models/join.sh 2-Forge-gpt-oss-20b
```

`join.sh` glues the parts back together, checks every file against `SHA256SUMS`, and puts the model in `~/.lmstudio/models/aran/<folder>`.

## Running the app

1. Install [LM Studio](https://lmstudio.ai) and download one of the models above.
2. Create a Python venv at `~/LocalVoice/.venv` with `websockets` (and `kokoro-onnx` for voice), or edit the paths in `app/main.swift` and `server.py`.
3. `python3 tools/pin_weights.py` keeps loaded model weights in RAM, so an idle model doesn't wear the SSD with swap.
4. `~/LocalVoice/.venv/bin/python server.py`, then open http://localhost:4480.

The `relay/` folder is an optional Cloudflare Worker for reaching Rivet from a phone. It needs your own `CONNECT_SECRET`; none is included.

`quant/` holds the scripts used to build the tuned 3-bit models.

## Licences

Code: MIT. Model weights keep their own Apache-2.0 licences (Gemma 4, gpt-oss, Qwen3.5); see each model folder's README.
