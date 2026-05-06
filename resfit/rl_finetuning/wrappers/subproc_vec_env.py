"""Common SubprocVecEnv infrastructure for all MuJoCo tasks.

Provides SubprocVecEnvMixin that handles:
- Worker process spawning (multiprocessing.spawn context)
- Pipe-based command dispatch (step_batch, reset_batch, get_qpos, close)
- qpos/qvel sync from workers to main process (for rendering)
- Graceful worker shutdown

Each task provides its own top-level worker loop function that handles
task-specific physics (init, step, reset, reward). The worker loop
must follow this pipe protocol:

    recv: ("step_batch", [(local_idx, pos3, quat4, grip), ...])
    send: [(state, obj_state, reward, terminated, truncated, *extra), ...]

    recv: ("reset_batch", [local_idx, ...])
    send: "ok"

    recv: ("get_qpos", local_idx)
    send: (qpos_copy, qvel_copy, extra_state_dict)

    recv: ("close",)
    send: None  (then exit)

Usage in VecEnv:
    class MuJoCoVecEnvFoo(SubprocVecEnvMixin):
        def __init__(self, ..., parallel_envs=True, num_workers=8):
            ...
            if parallel_envs and num_envs > 1:
                init_kwargs_fn = lambda env_indices: {...}  # task-specific
                self._init_parallel(
                    num_envs, num_workers,
                    worker_fn=_foo_worker_loop,
                    init_kwargs_fn=init_kwargs_fn,
                )
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable

import numpy as np


class SubprocVecEnvMixin:
    """Mixin providing parallel env stepping via subprocess workers.

    Subclass must call _init_parallel() to set up workers, then use
    _parallel_step(), _parallel_reset(), etc. in step()/reset().

    Attributes set by _init_parallel():
        _parallel: bool
        _workers: list[mp.Process]
        _worker_pipes: list[mp.Connection]
        _env_to_worker: dict[int, tuple[int, int]]  # global_idx → (worker_idx, local_idx)
    """

    def _init_parallel(
        self,
        num_envs: int,
        num_workers: int,
        worker_fn: Callable,
        init_kwargs_fn: Callable[[list[int]], dict],
    ) -> None:
        """Spawn worker processes.

        Args:
            num_envs: total number of environments
            num_workers: desired number of worker processes (capped to num_envs)
            worker_fn: top-level function for the worker loop (must be picklable)
            init_kwargs_fn: callable(env_indices) → dict of init kwargs for that worker
        """
        self._parallel = True
        self._workers: list[mp.Process] = []
        self._worker_pipes: list[mp.connection.Connection] = []
        self._env_to_worker: dict[int, tuple[int, int]] = {}
        self._needs_qpos_sync = False

        ctx = mp.get_context("spawn")
        actual_workers = min(num_workers, num_envs)

        # Distribute envs round-robin across workers
        envs_per_worker: list[list[int]] = [[] for _ in range(actual_workers)]
        for i in range(num_envs):
            envs_per_worker[i % actual_workers].append(i)

        for w_idx in range(actual_workers):
            env_indices = envs_per_worker[w_idx]
            for local_i, global_i in enumerate(env_indices):
                self._env_to_worker[global_i] = (w_idx, local_i)

            parent_pipe, child_pipe = ctx.Pipe()
            kwargs = init_kwargs_fn(env_indices)
            kwargs["num_local_envs"] = len(env_indices)

            proc = ctx.Process(
                target=worker_fn,
                args=(child_pipe, kwargs),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self._workers.append(proc)
            self._worker_pipes.append(parent_pipe)

        task_name = getattr(self, '__class__', type(self)).__name__
        print(f"[{task_name}] Spawned {actual_workers} workers for {num_envs} envs")

    def _parallel_step(
        self,
        actions_np: np.ndarray,
        num_envs: int,
    ) -> list[tuple]:
        """Send step_batch to all workers and collect results.

        Args:
            actions_np: (num_envs, 8) array [pos3, quat4, grip1]
            num_envs: total number of environments

        Returns:
            List of (global_env_idx, result_tuple) in order.
        """
        n_workers = len(self._worker_pipes)
        worker_batches: list[list[tuple]] = [[] for _ in range(n_workers)]
        worker_env_order: list[list[int]] = [[] for _ in range(n_workers)]

        for i in range(num_envs):
            w_idx, local_idx = self._env_to_worker[i]
            action = actions_np[i]
            worker_batches[w_idx].append(
                (local_idx, action[:3].copy(), action[3:7].copy(), float(action[7]))
            )
            worker_env_order[w_idx].append(i)

        # Send all batches (non-blocking from main's perspective)
        for w_idx in range(n_workers):
            if worker_batches[w_idx]:
                self._worker_pipes[w_idx].send(("step_batch", worker_batches[w_idx]))

        # Collect results, map back to global indices
        all_results: list[tuple | None] = [None] * num_envs
        for w_idx in range(n_workers):
            if not worker_batches[w_idx]:
                continue
            pipe = self._worker_pipes[w_idx]
            if not pipe.poll(timeout=300):
                raise RuntimeError(
                    f"Worker {w_idx} did not respond within 300s. "
                    f"Process alive: {self._workers[w_idx].is_alive()}"
                )
            results = pipe.recv()
            for j, global_i in enumerate(worker_env_order[w_idx]):
                all_results[global_i] = results[j]

        return all_results  # type: ignore[return-value]

    def _parallel_reset_all(self, num_envs: int) -> None:
        """Reset all envs across all workers."""
        n_workers = len(self._worker_pipes)
        worker_local_envs: list[list[int]] = [[] for _ in range(n_workers)]
        for i in range(num_envs):
            w_idx, local_idx = self._env_to_worker[i]
            worker_local_envs[w_idx].append(local_idx)
        for w_idx in range(n_workers):
            if worker_local_envs[w_idx]:
                self._worker_pipes[w_idx].send(("reset_batch", worker_local_envs[w_idx]))
        for w_idx in range(n_workers):
            if worker_local_envs[w_idx]:
                self._worker_pipes[w_idx].recv()

    def _parallel_reset_envs(self, env_ids: list[int]) -> None:
        """Reset specific envs."""
        n_workers = len(self._worker_pipes)
        worker_resets: list[list[int]] = [[] for _ in range(n_workers)]
        for eid in env_ids:
            w_idx, local_idx = self._env_to_worker[eid]
            worker_resets[w_idx].append(local_idx)
        for w_idx in range(n_workers):
            if worker_resets[w_idx]:
                self._worker_pipes[w_idx].send(("reset_batch", worker_resets[w_idx]))
        for w_idx in range(n_workers):
            if worker_resets[w_idx]:
                self._worker_pipes[w_idx].recv()

    def _parallel_get_qpos(self, env_idx: int) -> tuple:
        """Get qpos/qvel from worker for a single env."""
        w_idx, local_idx = self._env_to_worker[env_idx]
        self._worker_pipes[w_idx].send(("get_qpos", local_idx))
        return self._worker_pipes[w_idx].recv()

    def _sync_qpos_all(self, envs: list[dict], num_envs: int) -> None:
        """Batch-sync qpos/qvel from all workers to main-process render mirrors.

        Uses get_qpos_all command to fetch all envs from each worker in one
        round trip (instead of N sequential get_qpos calls).
        Falls back to per-env get_qpos if worker doesn't support get_qpos_all.
        """
        import mujoco
        n_workers = len(self._worker_pipes)

        # Build per-worker local index lists
        worker_local_indices: list[list[int]] = [[] for _ in range(n_workers)]
        worker_global_indices: list[list[int]] = [[] for _ in range(n_workers)]
        for i in range(num_envs):
            w_idx, local_idx = self._env_to_worker[i]
            worker_local_indices[w_idx].append(local_idx)
            worker_global_indices[w_idx].append(i)

        # Send batch requests
        for w_idx in range(n_workers):
            if worker_local_indices[w_idx]:
                self._worker_pipes[w_idx].send(("get_qpos_all",))

        # Collect and apply
        for w_idx in range(n_workers):
            if not worker_local_indices[w_idx]:
                continue
            all_qpos_data = self._worker_pipes[w_idx].recv()
            # all_qpos_data: list of (qpos, qvel, extra) for each local env
            for local_idx, global_idx in zip(
                worker_local_indices[w_idx], worker_global_indices[w_idx]
            ):
                qpos, qvel, extra = all_qpos_data[local_idx]
                env = envs[global_idx]
                env["data"].qpos[:] = qpos
                env["data"].qvel[:] = qvel
                if isinstance(extra, dict):
                    # Sync bowl position (model.body_pos, not in qpos)
                    if "bowl_pos" in extra and "bowl_body_id" in env:
                        bid = env["bowl_body_id"]
                        if bid >= 0:
                            env["model"].body_pos[bid] = extra["bowl_pos"]
                    # Sync cube/bowl init positions for bookkeeping
                    if "cube_pos_init" in extra:
                        env["cube_pos_init"] = extra["cube_pos_init"]
                    if "bowl_pos_init" in extra:
                        env["bowl_pos_init"] = extra["bowl_pos_init"]
                    # Sync grasp state
                    if "grasp_state" in env:
                        gs_keys = {"grasped", "contact_count"}
                        gs_update = {k: v for k, v in extra.items() if k in gs_keys}
                        if gs_update:
                            env["grasp_state"].update(gs_update)
                mujoco.mj_forward(env["model"], env["data"])

    def _close_workers(self) -> None:
        """Gracefully shut down all worker processes."""
        for pipe in self._worker_pipes:
            try:
                pipe.send(("close",))
                pipe.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
        for proc in self._workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._workers.clear()
        self._worker_pipes.clear()
