import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from tqdm import tqdm

try:
    from sae_lens.sae import SAE, SAEConfig
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "sae_lens is required for SegCompass-compatible SAE training. "
        "Install it or copy the SegCompass sae_lens package into this repo."
    ) from exc


def init_dist():
    if "RANK" not in os.environ:
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


class CachedActDataset(Dataset):
    def __init__(self, cached_dir):
        self.cached_dir = Path(cached_dir)
        self._cached_files = sorted(
            p for p in self.cached_dir.rglob("*.npz") if p.is_file() and p.name != "zero_sample.npz"
        )
        if not self._cached_files:
            raise FileNotFoundError(f"no cached .npz hidden files found under {self.cached_dir}")
        zero_path = self.cached_dir / "zero_sample.npz"
        if not zero_path.exists():
            first = self._cached_files[0]
            with np.load(first) as npz:
                hs = torch.zeros_like(torch.as_tensor(npz["hidden_states"], dtype=torch.bfloat16))
                am = torch.zeros_like(torch.as_tensor(npz["attention_mask"], dtype=torch.int64))
        else:
            with np.load(zero_path) as npz:
                hs = torch.as_tensor(npz["hidden_states"], dtype=torch.bfloat16)
                am = torch.as_tensor(npz["attention_mask"], dtype=torch.int64)
        self.zero_sample = {"hidden_states": hs, "attention_mask": am}

    def __len__(self):
        return len(self._cached_files)

    def __getitem__(self, idx):
        path = self._cached_files[idx]
        try:
            with np.load(path) as npz:
                hs = torch.as_tensor(npz["hidden_states"], dtype=torch.bfloat16)
                am = torch.as_tensor(npz["attention_mask"], dtype=torch.int64)
            return {"hidden_states": hs, "attention_mask": am}
        except Exception:
            print(f"[warning] bad sample -> {path}, fallback zero_sample")
            return self.zero_sample.copy()


def collate(batch):
    return {
        "hidden_states": torch.stack([b["hidden_states"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
    }


class SAETrainer:
    def __init__(self, config):
        self.config = config
        self.rank, self.world_size, self.local_rank = init_dist()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.optimizer = None
        self.lr_scheduler = None

    def _build_sae(self, trainable=True):
        cfg_dict = OmegaConf.to_container(self.config.sae_model, resolve=True)
        sae_cfg = SAEConfig.from_dict(cfg_dict)
        self.sae = SAE(sae_cfg).to(self.device)
        for p in self.sae.parameters():
            p.requires_grad_(trainable)
        self.sae.train(trainable)
        if self.world_size > 1:
            self.sae = DDP(
                self.sae,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        return self.sae

    @property
    def sae_module(self):
        return self.sae.module if isinstance(self.sae, DDP) else self.sae

    def _build_dataloader(self):
        dataset = CachedActDataset(self.config.data.cached_dir)
        n_valid = max(1, int(len(dataset) * 0.05)) if len(dataset) > 1 else 0
        n_train = len(dataset) - n_valid
        generator = torch.Generator()
        generator.manual_seed(int(self.config.data.split_seed))
        if n_valid > 0:
            train_set, valid_set = random_split(dataset, [n_train, n_valid], generator=generator)
        else:
            train_set, valid_set = dataset, dataset

        train_sampler = DistributedSampler(
            train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=False
        ) if self.world_size > 1 else None
        valid_sampler = DistributedSampler(
            valid_set, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
        ) if self.world_size > 1 else None

        self.train_loader = DataLoader(
            train_set,
            batch_size=int(self.config.data.batch_size),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=int(self.config.data.num_workers),
            pin_memory=True,
            persistent_workers=int(self.config.data.num_workers) > 0,
            collate_fn=collate,
            drop_last=False,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=int(self.config.data.batch_size),
            sampler=valid_sampler,
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collate,
            drop_last=False,
        )
        return self.train_loader, self.valid_loader

    def _build_optimizer(self):
        base_lr = float(self.config.optim.lr)
        self.optimizer = torch.optim.AdamW(self.sae.parameters(), lr=base_lr)
        total_steps = int(self.config.train.max_epochs) * len(self.train_loader) + 1
        warmup_steps = int(total_steps * float(self.config.optim.warmup_ratio))
        cosine_steps = total_steps - warmup_steps
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=base_lr * float(self.config.optim.eta_min_factor),
        )
        if warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
            self.lr_scheduler = SequentialLR(self.optimizer, [warmup, cosine], [warmup_steps])
        else:
            self.lr_scheduler = cosine
        return self.optimizer, self.lr_scheduler

    def _compute_loss(self, batch):
        assert self.config.sae_model.architecture == "standard", "Only support SegCompass standard SAE here."
        x = batch["hidden_states"].to(device=self.device, dtype=next(self.sae.parameters()).dtype)
        mask = batch["attention_mask"].to(self.device)
        valid = (mask != 0).float()

        sparse_embeds, x_recon = self.sae(x)
        se_token = F.mse_loss(x_recon, x, reduction="none").sum(-1)
        mse_loss = (se_token * valid).sum() / valid.sum().clamp_min(1)

        weighted_sparse_embeds = sparse_embeds * self.sae_module.W_dec.norm(dim=1)
        sparsity = weighted_sparse_embeds.norm(p=1, dim=-1)
        l1_loss = (sparsity * valid).sum() / valid.sum().clamp_min(1)
        loss = mse_loss + float(self.config.loss.l1_coef) * l1_loss
        return mse_loss, l1_loss, loss

    def _save_checkpoint(self):
        if self.rank != 0:
            return None
        save_dir = Path(self.config.ckpt.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"ep_{self.epoch}.pt"
        ckpt = {
            "sae": {k: v.cpu() for k, v in self.sae_module.state_dict().items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "epoch": self.epoch,
        }
        torch.save(ckpt, path)
        return str(path)

    def _load_checkpoint(self, load_path):
        if not load_path:
            return None
        load_path = Path(load_path)
        ep_re = re.compile(r"ep_(\d+)\.pt$")
        if load_path.is_dir():
            candidates = []
            for path in load_path.glob("ep_*.pt"):
                match = ep_re.search(path.name)
                if match:
                    candidates.append((int(match.group(1)), path))
            if not candidates:
                return None
            _, load_path = max(candidates, key=lambda item: item[0])
        ckpt = torch.load(load_path, map_location="cpu")
        self.sae_module.load_state_dict(ckpt["sae"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            self.lr_scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = int(ckpt.get("epoch", 0))
        return str(load_path)

    def _write_log(self, log, epoch=None, step=None):
        if self.rank != 0:
            return
        itr_name = "Step" if step is not None else "Epoch"
        itr_point = step if step is not None else epoch
        log_path = Path(self.config.ckpt.save_dir) / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"======== {itr_name} {itr_point} ========\n")
            f.write(json.dumps(log, ensure_ascii=False, sort_keys=True, indent=2) + "\n\n")

    def _reduce_metrics(self, metrics):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= self.world_size
        return metrics

    def _train_loop(self, epoch):
        train_metrics = []
        self.sae.train()
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        iterator = self.train_loader if self.rank != 0 else tqdm(self.train_loader, desc=f"Train Epoch: {epoch}")
        for step, batch in enumerate(iterator):
            self.optimizer.zero_grad()
            mse_loss, l1_loss, loss = self._compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()

            metrics = torch.stack([mse_loss.detach(), l1_loss.detach(), loss.detach()]).to(self.device)
            metrics = self._reduce_metrics(metrics)
            metric = {
                "mse_loss": metrics[0].cpu().item(),
                "l1_loss": metrics[1].cpu().item(),
                "loss": metrics[2].cpu().item(),
            }
            train_metrics.append(metric)
            self._write_log(metric, step=step + epoch * len(self.train_loader))
        return {k: sum(m[k] for m in train_metrics) / len(train_metrics) for k in train_metrics[0]}

    @torch.no_grad()
    def _validate(self, epoch):
        self.sae.eval()
        valid_metrics = []
        iterator = self.valid_loader if self.rank != 0 else tqdm(self.valid_loader, desc=f"Valid Epoch: {epoch}")
        for batch in iterator:
            mse_loss, l1_loss, loss = self._compute_loss(batch)
            metrics = torch.stack([mse_loss.detach(), l1_loss.detach(), loss.detach()]).to(self.device)
            metrics = self._reduce_metrics(metrics)
            valid_metrics.append(
                {
                    "mse_loss": metrics[0].cpu().item(),
                    "l1_loss": metrics[1].cpu().item(),
                    "loss": metrics[2].cpu().item(),
                }
            )
        log = {k: sum(m[k] for m in valid_metrics) / len(valid_metrics) for k in valid_metrics[0]}
        self._write_log(log, epoch=epoch)
        return log

    def fit(self):
        self._build_dataloader()
        self._build_sae()
        self._build_optimizer()
        self._load_checkpoint(self.config.ckpt.load_path)
        if Path(self.config.ckpt.save_dir).is_dir():
            self._load_checkpoint(self.config.ckpt.save_dir)
        start_epoch = self.epoch
        for epoch in range(start_epoch, int(self.config.train.max_epochs)):
            self.epoch = epoch
            self._train_loop(epoch)
            self._validate(epoch)
            self._save_checkpoint()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def load_cfg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", required=True)
    args, rest = parser.parse_known_args()
    yaml_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(rest)
    return OmegaConf.merge(yaml_cfg, cli_cfg)


def main():
    trainer = SAETrainer(load_cfg())
    trainer.fit()


if __name__ == "__main__":
    main()

