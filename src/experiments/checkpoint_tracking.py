"""Post-update checkpoint bookkeeping; no validation or test evaluation."""
import hashlib
from pathlib import Path
from .configuration import require, digest
from .training_engine import write_json


def schedule(job, integration):
    return [1024, 2048] if integration else list(job['checkpoints'])


def parameter_hash(policy):
    h = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        for part in (name.encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()):
            h.update(len(part).to_bytes(8, 'big')); h.update(part)
    return h.hexdigest()


class Tracker:
    def __init__(self, out, targets, plan_hash, job_id):
        require(targets == sorted(set(targets)) and bool(targets) and targets[0] > 0, 'checkpoint schedule')
        self.out = Path(out); self.targets = targets; self.rows = []
        self.plan_hash = plan_hash; self.job_id = job_id

    def capture(self, model, transitions, unique_episodes):
        step = int(model.num_timesteps)
        if len(self.rows) == len(self.targets): return
        target = self.targets[len(self.rows)]
        require(step <= target, 'missed checkpoint')
        if step != target: return
        require(step == transitions, 'checkpoint transition mismatch')
        folder = self.out / 'checkpoints' / f'step_{step:08d}'
        folder.mkdir(parents=True, exist_ok=False)
        model.save(folder/'model')
        ph = parameter_hash(model.policy)
        loaded = type(model).load(folder/'model.zip', device='cpu')
        require(loaded.num_timesteps == step and parameter_hash(loaded.policy) == ph, 'checkpoint reload mismatch')
        row = dict(step=step, job_id=self.job_id, plan_hash=self.plan_hash, unique_train_episodes=unique_episodes, model_sha256=digest(folder/'model.zip'), parameter_sha256=ph, path=(folder/'model.zip').relative_to(self.out).as_posix())
        write_json(folder/'checkpoint.json', row); self.rows.append(row)

    def finish(self):
        require([r['step'] for r in self.rows] == self.targets, 'incomplete checkpoints')
        for r in self.rows:
            require(digest(self.out/r['path']) == r['model_sha256'], 'checkpoint file changed')
        write_json(self.out/'checkpoints.json', self.rows)
