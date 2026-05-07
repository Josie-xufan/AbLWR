import torch
import os

class ModelCheckpoint_val:
    def __init__(self, monitor, mode='min', save_top_k=1, dirpath='./', filename='best', logger=None, verbose=True):
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.dirpath = dirpath
        self.filename = filename
        self.verbose = verbose
        self.best_score = None
        self.logger = logger
        self.saved_ckpts = []

    def step(self, model, metrics, epoch):
        score = metrics
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == 'min' and score < self.best_score:
            improved = True
        elif self.mode == 'max' and score > self.best_score:
            improved = True

        if improved:
            self.best_score = score
            ckpt_path = f"{self.dirpath}/{self.filename}_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            self.saved_ckpts.append(ckpt_path)
            if self.verbose:
                self.logger.info(f"[ModelCheckpoint] Saved improved checkpoint at epoch {epoch}, score={score}")

            # Keep only top k
            if self.save_top_k > 0:
                if len(self.saved_ckpts) > self.save_top_k:
                    to_remove = self.saved_ckpts.pop(0)
                    try:
                        os.remove(to_remove)
                    except Exception:
                        pass



class EarlyStopping:
    def __init__(self, monitor, patience=10, min_delta=0., mode='min', logger=None, verbose=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.num_bad_epochs = 0
        self.logger = logger
        self.should_stop = False

    def step(self, metrics):
        score = metrics
        if self.best_score is None:
            self.best_score = score
            self.num_bad_epochs = 0
        else:
            if self.mode == 'min':
                if score < self.best_score - self.min_delta:
                    self.best_score = score
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1
            elif self.mode == 'max':
                if score > self.best_score + self.min_delta:
                    self.best_score = score
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
            if self.verbose:
                self.logger.info(f"[EarlyStopping] Triggered at epoch (patience={self.patience})")


class LearningRateMonitor:
    def __init__(self):
        self.history = []

    def step(self, optimizer, epoch, batch_idx):
        lrs = [group['lr'] for group in optimizer.param_groups]
        self.history.append({'epoch': epoch, 'batch_idx': batch_idx, 'lrs': lrs})

    def log_last(self):
        if self.history:
            print(f"[LearningRateMonitor] Epoch {self.history[-1]['epoch']} batch_idx {self.history[-1]['batch_idx']} LR: {self.history[-1]['lrs']}")



