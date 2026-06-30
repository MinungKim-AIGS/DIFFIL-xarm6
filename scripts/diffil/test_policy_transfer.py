#!/usr/bin/env python3
"""Policy-transfer fidelity test — server-side, NO robot, NO network.

Verifies the last untested seam of the pipeline: a TensorFlow SAC actor (what the
learner trains and exports) produces the SAME deterministic actions after going
through the full transfer path

    weight_io.export_actor  ->  comm.encode/decode_weights (msgpack wire)
                            ->  policy_runtime.NumpyActor   (TF-free laptop)

plus the version hot-swap logic (newer weights apply, stale are ignored).

Why this matters: the laptop runs the policy as a pure-numpy `tanh(mean)` forward.
If that doesn't numerically match the TF actor, the robot would execute a different
policy than the one the server trained. This test catches any such drift.

Run from the DIFF-IL repo folder (where sac_models.py + our diffil files live):
    python test_policy_transfer.py

Expects: max |TF - numpy| ~ 1e-6  (well under the 1e-4 pass threshold).
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 21
ACT_DIM = 6
TOL = 1e-4


def build_tf_actor(seed: int = 0):
    """A StochasticActor with the SAME architecture build_diffil.make_actor uses."""
    import tensorflow as tf
    from sac_models import StochasticActor
    tf.random.set_seed(seed)
    layers = [
        tf.keras.layers.Dense(256, "relu", kernel_initializer="orthogonal"),
        tf.keras.layers.Dense(256, "relu", kernel_initializer="orthogonal"),
        tf.keras.layers.Dense(ACT_DIM * 2,
                              kernel_initializer=tf.keras.initializers.Orthogonal(0.01)),
    ]
    actor = StochasticActor(layers)                 # no obs-norm (matches the pipeline)
    actor.get_action(tf.zeros([1, OBS_DIM], tf.float32), 0.0)   # call once -> builds weights
    return actor


def main():
    import tensorflow as tf
    import weight_io
    from comm import encode_weights, decode_weights
    from policy_runtime import NumpyActor

    print("=" * 56)
    print("POLICY TRANSFER FIDELITY TEST")
    print("=" * 56)

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((128, OBS_DIM)).astype(np.float32)

    # ---- 1) numerical equivalence TF actor vs transferred numpy actor ----
    actor = build_tf_actor(seed=0)
    tf_act = actor.get_action(tf.constant(obs), 0.0).numpy()     # deterministic = tanh(mean)

    w = weight_io.export_actor(actor, version=1)
    w_wire = decode_weights(encode_weights(w))                   # exercise the msgpack wire format
    npy = NumpyActor(w_wire)
    np_act = npy.get_action(obs, 0.0)

    err = float(np.max(np.abs(tf_act - np_act)))
    print(f"[1] act_dim={w['act_dim']} layers={len(w['layers'])} "
          f"acts={[L['act'] for L in w['layers']]}")
    print(f"    max |TF - numpy| over 128 obs = {err:.2e}   "
          f"-> {'OK' if err < TOL else 'FAIL'}")

    # also confirm both are valid actions in [-1, 1]
    rng_ok = bool(np.all(np.abs(np_act) <= 1.0 + 1e-6))
    print(f"    numpy actions within [-1,1]: {rng_ok}")

    # ---- 2) version hot-swap logic ----
    a = NumpyActor(decode_weights(encode_weights(
        weight_io.export_actor(build_tf_actor(seed=1), version=1))))
    applied_newer = a.update_weights(decode_weights(encode_weights(
        weight_io.export_actor(build_tf_actor(seed=2), version=2))))
    applied_stale = a.update_weights(decode_weights(encode_weights(
        weight_io.export_actor(build_tf_actor(seed=3), version=1))))
    print(f"[2] hot-swap: newer(v2) applied={applied_newer} (expect True), "
          f"stale(v1) applied={applied_stale} (expect False), now at v{a.version}")

    ok = (err < TOL) and rng_ok and applied_newer and (not applied_stale)
    print("=" * 56)
    print("RESULT:", "ALL OK  — policy transfer is faithful." if ok else "FAIL")
    print("=" * 56)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
