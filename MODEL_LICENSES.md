# Model licenses

`scripts/fetch_models.sh` downloads these files from their publishers. They are not stored in this repository.

| Model | Used for | License | Source |
|---|---|---|---|
| Winnow-12B Q8_0 (EldanRing, fine-tune of Google Gemma 4 12B IT) | sentiment (Customer Experience use case) | Apache-2.0 | https://huggingface.co/EldanRing/Winnow-12B |
| nemotron-speech-streaming-en-0.6b Q8_0 (NVIDIA) | speech recognition | NVIDIA Open Model License | https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b |
| Nemotron-3-Diarization (NVIDIA), Q8_0 GGUF and safetensors | who spoke when | OpenMDW-1.1 | https://huggingface.co/nvidia/Nemotron-3-Diarization |

Runtimes, from GitHub:

| Runtime | License | Source |
|---|---|---|
| NeMo-Speech.cpp v0.2.0 (prebuilt CUDA archive) | Apache-2.0 | https://github.com/NVIDIA/NeMo-Speech.cpp |
| winnow-inference @ a50c7d0 (built from source; includes llama.cpp) | MIT | https://github.com/EldanRing/winnow-inference |

All three model licenses allow use and redistribution, provided their license text and notices travel with any
copy you make. If you mirror the files (for example to S3 for `MODEL_BASE_URL`), keep each license next to them.
