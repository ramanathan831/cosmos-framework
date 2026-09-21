# Cosmos inference services

Use the existing `cosmos3-inference` skill. These services are Framework-owned
adapters; setup, approval, job records, and platform monitoring remain in
`../execution/`. Docker, SLURM, Kubernetes, Brev, and virtualenv are available
when their GPU/network prerequisites fit the selected service.

## Reasoner (Cosmos3 Nano / Edge and compatible Qwen checkpoints)

Install Framework's `workflows` extra into the selected GPU environment. Prepare
a complete Hugging Face export first: use the reasoner checkpoint helpers for
DCP and PEFT inputs. Do not point the HTTP server at an unmerged adapter.

```bash
cosmos-reasoner-serve --model-path /models/reasoner --model-name cosmos \
  --media-root /data --host 127.0.0.1 --port 8080
```

`/health`, `/info`, `/v1/models`, `/infer`, and
`/v1/chat/completions` are implemented locally in
`cosmos_framework/inference/reasoner/service.py`. Chat requests accept text,
`image_url`, and `video_url` content parts, including base64 data URIs.
Only explicitly allowed media roots are readable. Remote media requires
`--allow-remote-media` and trusted callers (it grants outbound network access).
Requests are serialized; streaming is rejected. Sampling and vision options
are passed through to the native reasoner runtime.

## Predict 2.5 (separate from PAIDF Predict augmentation)

Use a clean, revision-pinned checkout of
[the official Predict source](https://github.com/nvidia-cosmos/cosmos-predict2.5)
and its own installed GPU environment. The model implementation stays upstream;
the HTTP adapter and process lifecycle live in Framework.

```bash
cosmos-predict-serve --predict-root /opt/cosmos-predict2.5 \
  --predict-commit <full-commit-sha> --predict-python /opt/predict/bin/python \
  --output-dir /outputs --model 2B/post-trained --media-root /data
```

The adapter executes upstream `examples/inference.py` in a fresh process for
each request. This preserves native guardrails, checkpoint selection,
offloading, autoregressive options, and context parallelism, at the cost of
reloading the model. `--context-parallel-size N` uses torchrun for N GPUs.
Guardrails remain enabled unless explicitly disabled. Requests time out after
`--timeout` seconds; the full process group is terminated on timeout.

For multiview, select `--multiview --model 2B/auto/multiview` and send
`{"sample_path": "/data/native-multiview-input.json"}`. This uses upstream
`examples/multiview.py` and its native input schema. The JSON and all referenced
media must be prepared by a trusted operator in the mounted data area.
Base generation accepts `text2world`, `image2world`, and `video2world`; responses
contain MP4 data URIs. Generated files and per-request logs remain under the
configured output directory. Model keys come from the pinned upstream checkout,
not an invented registry mapping.

`docker/predict-service.Dockerfile` extends a user-built native Predict image;
`cosmos-predict:local` is a local build target, not a published image.

## Annotation LLM/VLM endpoint

The annotation clients consume standard OpenAI-compatible endpoints.
For an explicitly chosen compatible model, use the existing vLLM environment:

```bash
vllm serve <model-id-or-local-path> --host 127.0.0.1 --port 8080
```

This is an optional native model server, not a Framework training dependency.
Verify the selected model's multimodal support and memory requirements before
deployment. The annotation YAML chooses an explicit model and base URL; no
remote API model is silently selected.

## Deployment and safety

Bind to loopback by default. If exposing a container port, bind inside the
container to `0.0.0.0` and publish only to a controlled host interface. Configure
authentication, TLS, request limits, and a restrictive egress policy at the
ingress before exposing any endpoint to untrusted clients. These adapters do
not implement authentication. Do not bake tokens into images or CLI arguments.

Preview the concrete image/environment, model, source revisions, mounts,
ports, GPU allocation, estimated runtime, and output location before launch.
Use the selected platform's four verbs; do not infer liveness from job records.
CPU request-contract tests do not establish GPU/model compatibility.
