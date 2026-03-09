import multiprocessing as mp
import numpy as np

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    # Import inside worker to ensure each process has it
    from oskp_rl_up_buffer_experiments_v3 import proxy_scores_for_heuristics

    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'new_box_arrival':
                state = env.new_box_arrival(data)
                remote.send(state)
            elif cmd == 'get_proxy_scores':
                # data is pred_mask_probs
                scores = proxy_scores_for_heuristics(
                    env.current_height_map, env.current_box,
                    env.pallet_size, env.max_height, data
                )
                remote.send(scores)
            elif cmd == 'choose_action_by_heuristic':
                heuristic, pred_mask = data
                action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask)
                remote.send((action, mapping))
            elif cmd == 'get_env_info':
                info = {
                    'invalid_learned': env.invalid_actions_learned,
                    'invalid_attempted': env.invalid_actions_attempted,
                    'placed_boxes_count': len(env.placed_boxes),
                    'placed_boxes': env.placed_boxes.copy() if data.get('full', False) else None
                }
                remote.send(info)
            else:
                raise NotImplementedError(f"Command {cmd} not recognized")
    except EOFError:
        pass
    except Exception as e:
        import traceback
        print(f"Worker Error: {e}")
        traceback.print_exc()

class SubprocVecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.array(rews), np.array(dones), infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
