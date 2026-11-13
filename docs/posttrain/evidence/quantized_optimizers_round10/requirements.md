# Native action-policy integration — prospective implementation contract

Starting source: 2516e9f1e9aecb0b68d6a21e27dd835f456109ab. Prior WTS results and thresholds remain unchanged. No aggregate resource ceiling applies; at most two owned GPU allocations, configured routing, finite hypotheses and terminal accounting remain required.

Dependency map: SFT -> build_optimizer -> existing ordinary FP32 bnb owner; GR00T Joint PPO -> model-owned actor/value partition -> build_optimizer -> explicit local-master owner; AlpaGym closed-loop -> AlpamayoTrainModel.setup -> native shared factory with preserved AlpaGym hyperparameters. The GR00T native owner is FSDP2. AlpaGym external setup must still be verified against the available runtime before its strategy can be claimed.

Implementation sequence: (1) shared FP32 local-master adapter and same-topology DCP payload; (2) native GR00T construction, actor/value ownership and completed-window recovery; (3) AlpaGym construction and completed-collection boundary recovery. Each integration retains native rollout, reward, processing and policy publication. No surrogate task qualifies either model.

Precision: retain BF16 live model parameters and existing forward/reduction/transfer policy. Copy already reduced/clipped local gradients to ordinary FP32 masters; apply one backend update; copy rounded values into the currently owned model shard. One-dimensional FSDP2 Shard(0) with complete learner rank group only initially. Quantization blocks are local-shard blocks. Explicit FP32-state exceptions use logical names. Masters and staging are separately accounted; legacy/default routes are unchanged.

Checkpoint: native atomic transaction owns model/control/data. Optimizer component contains per-rank opaque payloads of masters, moments, scales/maps, lazy state, groups and exact layout identity. Generic torch optimizer conversion must never reinterpret quantization metadata. Stage before mutation, converge failures, roll back application failures. Same topology only. No inference dependency is added.

First accelerator gate (infrastructure, not model qualification): two CUDA ranks, BF16 FSDP2, native LLMTrainer accumulation/clipping/scheduler, fused FP32-state control plus nonpaged/paged eight-bit; eight updates and fresh-process continuation from four; compare every resumed input/update/model/master/moment/control state, frozen exclusion, empty unused shard and one-rank nonfinite rejection. Model-specific closed-loop development and prospective final protocols follow only after this gate and asset/runtime validation.
