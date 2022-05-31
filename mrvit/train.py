from pytorch_lightning import LightningModule
import pytorch_lightning
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from mrvit.config import Config
import torch
import typing
import mrvit.models as models
from mrvit.dataset import DataModule

class TrainHandler(LightningModule):

    _SUPPORTED_MODELS = {
        'EfficientNet':  models.efficientnet.__dict__.get('EfficientNet'),
        'MRNet': models.mrnet.__dict__.get('MRNet'),
        'SwinTransformer': models.swin_transformer.__dict__.get('SwinTransformer'),
        'VisionTransformer': models.vision_transformer.__dict__.get('VisionTransformer')
    }
    
    _SUPPORTED_LOSS = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss
    }

    _SUPPORTED_OPTIMIZERS = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }

    _SUPPORTED_SCHEDULERS = {
        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    }

    """
        TrainHandler

        Base class for handling all necessary functionality required during training and validation phase
        with the help of PytorchLightning framework

        Args:
        :config: Config - config instance with parameters to apply during instance initialization

    """

    def __init__(self,
                 config: Config):
        super().__init__()
        self._config = config
        self._build_model()
        self._build_criterion()
        self.save_hyperparameters(self._config.dict())
    
    def _build_model(self) -> typing.TypeVar('TorchModel'):
        if self._config.train.model_name not in self._SUPPORTED_MODELS:
            raise ValueError(f"{self._config.train.model_name} not supported, check your configuration or check supported models in `models` package")
        self._model = self._SUPPORTED_MODELS.get(self._config.train.model_name)(self._config)

    def _build_criterion(self) -> typing.TypeVar('TorchLoss'):
        if self._config.train.loss.name not in self._SUPPORTED_LOSS:
            raise ValueError(f"{self._config.train.loss.name} not supported, check your configuration")
        criterion: typing.Callable = self._SUPPORTED_LOSS.get(self._config.train.loss.name)
        self._criterion = criterion if self._config.train.loss.params is None else criterion(**self._config.train.loss.params)
    
    def _build_optimizer(self) -> typing.TypeVar('TorchOptimizer'):
        if self._config.train.optimizer.name not in self._SUPPORTED_OPTIMIZERS:
            raise ValueError(f"{self._config.train.optimizer.name} not supported, check your configuration")
        if self._config.train.optimizer.params is not None:
            self._optimizer = self._SUPPORTED_OPTIMIZERS[self._config.train.optimizer.name](self.parameters(), **self._config.train.optimizer.params)
        else:
            self._optimizer = self._SUPPORTED_OPTIMIZERS[self._config.train.optimizer.name](self.parameters())
    
    def _build_scheduler(self) -> typing.TypeVar('TorchScheduler'):
        if self._config.train.scheduler.name not in self._SUPPORTED_SCHEDULERS:
            raise ValueError(f"{self._config.train.scheduler.name} not supported, check your configuration")
        if self._config.train.scheduler.params is not None:
            self._scheduler = self._SUPPORTED_SCHEDULERS.get(self._config.train.scheduler.name)(self._optimizer, **self._config.train.scheduler.params)
        else:    
            self._scheduler = self._SUPPORTED_SCHEDULERS.get(self._config.train.scheduler.name)(self._optimizer)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(x)
        return out

    def training_step(self, batch: typing.Tuple[torch.Tensor, int, float], batch_idx: int):
        loss, pred, labels = self._share_step(batch, 'train')
        return {'loss': loss, 'pred': pred, 'labels': labels}

    def validation_step(self, batch: typing.Tuple[torch.Tensor, int, float], batch_idx: int):
        loss, pred, labels = self._share_step(batch, 'val')
        return {'loss': loss, 'pred': pred, 'labels': labels}

    def _share_step(self, batch: typing.Tuple[torch.Tensor, int], mode: str):
        image, label, weight = batch
        logit = self.forward(image).squeeze(0)
        loss = self._criterion(logit, label) if not callable(self._criterion) else self._criterion(pos_weight=weight)(logit, label)
        return loss, logit.sigmoid(), label.int()

    def training_epoch_end(self, outputs):
        self._share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self._share_epoch_end(outputs, 'val')
    
    def _share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        auc_metric = torchmetrics.functional.auroc(preds, labels)
        accuracy_metric = torchmetrics.functional.precision(preds, labels)
        recall_metric = torchmetrics.functional.recall(preds, labels)
        self.log(f'{mode}_auc', auc_metric, prog_bar=True)
        self.log(f'{mode}_accuracy', accuracy_metric, prog_bar=True)
        self.log(f'{mode}_recall', recall_metric, prog_bar=True)
    
    def configure_optimizers(self) -> typing.Dict[str, typing.TypeVar('torchObject')]:
        self._build_optimizer()
        self._build_scheduler()
        return {"optimizer": self._optimizer, "lr_scheduler": self._scheduler}

class TrainTriggerer:

    """
        TrainTriggerer

        Simple wrapper to trigger model training

        Args:
        :config: Config - config instance with parameters to apply during instance initialization

    """

    def __init__(self, config: Config):
        self._config = config
        self._handler = TrainHandler(self._config)

    def _prepare_session(self) -> None:
        """
            Prepare training session to squeeze best performance from given hardware
        """
        torch.autograd.set_detect_anomaly(True)
        if self._config.train.gpus > 0:
            torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)
        seed_everything(self._config.train.seed)

    def trigger(self) -> None:
        """
            trigger training session
        """
        self._prepare_session()
        datamodule = DataModule(self._config)
        early_stopping = EarlyStopping(monitor=self._config.train.monitor_metric,
                                       verbose=self._config.train.verbose,
                                       patience=self._config.train.patience,
                                       mode="max")
        lr_monitor = LearningRateMonitor()
        loss_checkpoint = ModelCheckpoint(
            filename=self._config.train.save_model_filename,
            monitor=self._config.train.monitor_metric,
            save_top_k=1,
            mode="max",
            save_last=False
        )

        logger = TensorBoardLogger(self._config.train.model_name)

        trainer = pytorch_lightning.Trainer(
            accelerator = 'gpu' if self._config.train.gpus else 'cpu',
            devices=self._config.train.gpus,
            accumulate_grad_batches=self._config.train.accumulate_grad_batches,
            fast_dev_run=self._config.train.fast_dev_run,
            resume_from_checkpoint=self._config.train.resume_from_checkpoint,
            num_sanity_val_steps=self._config.train.num_sanity_val_steps,
            logger=logger,
            max_epochs=self._config.train.epochs,
            callbacks=[lr_monitor, loss_checkpoint, early_stopping]
        )

        trainer.fit(self._handler, datamodule=datamodule)
