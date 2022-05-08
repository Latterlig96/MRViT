from pydantic import BaseModel, Field, validator
from pathlib import Path
from typing import List, ClassVar, Optional, Dict, Any

__all__ = ['Config',]

class AugmentationSettings(BaseModel):

    """
        AugmentationSettings

        Args:
        :image_size: List[int] - image size applied to images in preprocessing stage
        :rotate_angle: int - rotate angle applied to images during augmentation phase
        :image_mean: List[float] - image mean applied during image normalization 
        :image_std: List[float] - image standard deviation applied during image normalization

    """

    image_size: List[int] = Field(description="Input image size for given model")
    rotate_angle: int = Field(description="angle of rotation")
    image_mean: List[float] = Field(description="Image mean applied in normalization stage")
    image_std: List[float] = Field(description="Image standard deviation applied in normalization stage")

class ModelSettings(BaseModel):

    """
        ModelSettings

        Args:
        :use_pretrained: bool - whether to use pretrained model during training phase
        :num_classes: int - number of classes to output from pretrained model (see `timm` library documentation for more informations)
        :input_channels: int - number of input channels for pretrained model (see `timm` library documentation for more informations) 
        :output_dim: int - output dimension for model

    """

    use_pretrained: bool = Field(description="whether to use pretrained model during model training")
    num_classes: int = Field(description="number of classes during training phase")
    input_channels: int = Field(description="number of input channels for single image")
    output_dim: int = Field(description="output dimension from trained network")

class LossSettings(BaseModel):

    """
        LossSettings

        Args:
        :name: str - name of loss function applied during training phase
        :params: Dict[str, Any] - additional parameters to apply to loss function during instance initialization

    """

    name: str = Field(description="Name of the loss function to use")
    params: Optional[Dict[str, Any]] = Field(description="Additional arguments to pass to loss function instance, by default None", default=None)

class OptimizerSettings(BaseModel):

    """
        OptimizerSettings

        Args:
        :name: str - name of optimizer applied during training phase
        :params: Dict[str, Any] - additional parameters to apply to optimizer during instance initialization
        
    """

    name: str = Field(description="Name of the optimizer to use during training phase")
    params: Optional[Dict[str, Any]] = Field(description="Additional arguments to pass to optimizer instance, by default None", default=None)

class SchedulerSettings(BaseModel):

    """
        SchedulerSettings

        Args:
        :name: str - name of scheduler applied during training phase
        :params: Dict[str, Any] - additional parameters to apply to scheduler during instance initialization
        
    """

    name: str = Field(description="Name of the learning rate scheduler to use during training phase")
    params: Optional[Dict[str, Any]] = Field(description="Additional arguments to pass to scheduler instance, by default None", default=None)

class TrainSettings(BaseModel):

    _AVAILABLE_PLANES: ClassVar = ['axial', 'coronal', 'sagital']
    _AVAILABLE_TASKS: ClassVar = ['abnormal', 'acl', 'meniscus']

    """
        TrainSettings

        Args:
        :train: bool - information denoting if we are in training phase
        :root_dir: Path - root directory for data
        :plane: str - which plane (either ['axial', 'coronal', 'sagital']) to use during training
        :task: str - which task (either ['abnormal', 'acl', 'meniscus']) to use during training
        :model_name: str - model name that will be used to invoke model instance during training
        :save_mdoel_filename: str - filename in which the trained model will be saved
        :loss: LossSettings
        :optimizer: OptimizerSettings
        :scheduler: SchedulerSettings
        :batch_size: int - batch_size applied during training phase
        :epochs: int - number of epochs
        :num_workers: int - number of workers to use during data loading
        :seed: int - seed for random number generator
        :loss_monitor: str - which loss function to monitor during training
        :patience: int - if loss function won't decrease, specify number of patience epochs before stopping training session
        :verbose: bool - make training verbose
        :gpus: int - if specified, given number of GPU instances will be used during training
        :accumulate_grad_batches: int - Number of batch instance loss that will be accumulated before pyTorch `step` function will be invoked
        :fast_dev_run: bool - parameters denotes creating predefined train, val and test loaders will be created and invoked on model to debug all train/val 
                              phase to detect possible bugs
        :num_sanity_val_steps: int - Number of validation steps to be run before training to check if there is any errors during validation phase
        :resume_from_checkpoint: Optional[str] - Whether to use previously trained model and resume from given checkpoint

    """

    train: bool = Field(description="whether to run computation in train mode")
    root_dir: Path = Field(description="root dir where data is stored")
    plane: str = Field(description="specify which plane to use during training")
    task: str = Field(description="specify which case will be handled during training")
    model_name: str = Field(description="name of predefined model that is specified in models package")
    save_model_filename: str = Field(description="filename in which the trained model will be saved, by default `best_model`", default="best_model")
    loss: LossSettings
    optimizer: OptimizerSettings
    scheduler: SchedulerSettings
    batch_size: int = Field(description="batch size during training", gt=0)
    epochs: int = Field(description="number of epochs", gt=0)
    num_workers: int = Field(description="number of workers to use during data loading, by default 1", default=6)
    seed: int = Field(description="seed for random number generator, by default 47", default=47)
    monitor_metric: str = Field(description="which metric to monitor during training, by default `val_auc`", default="val_auc")
    patience: int = Field(description="if loss function won't decrease, specify number of patience epochs before stopping training session, by default 1", default=1)
    verbose: bool = Field(description="make training verbose, by default True", default=True)
    gpus: int = Field(description="if specified, given number of GPU instances will be used during training, by default 1", default=1)
    accumulate_grad_batches: int = Field(description="""Number of batch instance loss that will be accumulated before pyTorch `step` function will be invoked.
                                                        Especially useful when dealing with large models and low batch size, by default 1""", default=1)
    fast_dev_run: bool = Field(description="""if True, predefined train, val and test loaders will be created and invoked on model to debug all train/val 
                                              phase to detect possible bugs, by default False""", default=False)
    num_sanity_val_steps: int = Field(description="Number of validation steps to be run before training to check if there is any errors during validation phase, by default 0", default=0)
    resume_from_checkpoint: Optional[str] = Field(description="Whether to use previously trained model and resume from given checkpoint, by default None", default=None)

    @validator('plane')
    def check_if_plane_is_available(cls, value: str):
        if value not in cls._AVAILABLE_PLANES:
            raise ValueError(f"Plane must be either ['axial', 'coronal', 'sagital'] but got {value} instead")    
        return value
    
    @validator('task')
    def check_if_task_is_available(cls, value: str):
        if value not in cls._AVAILABLE_TASKS:
            raise ValueError(f"Task must be either ['abnormal', 'acl', 'meniscus'] but got {value} instead")    
        return value

class ValidationSettings(BaseModel):

    _AVAILABLE_PLANES: ClassVar = ['axial', 'coronal', 'sagital']
    _AVAILABLE_TASKS: ClassVar = ['abnormal', 'acl', 'meniscus']
    
    """
        ValidationSettings

        Args:

        :train: bool - information denoting if we are in training phase
        :root_dir: Path - root directory for data
        :plane: str - which plane (either ['axial', 'coronal', 'sagital']) to use during training
        :task: str - which task (either ['abnormal', 'acl', 'meniscus']) to use during training
        :batch_size: int - batch_size applied during training phase
        :num_workers: int - number of workers to use during data loading

    """

    train: bool = Field(description="whether to run computation in train mode")
    root_dir: Path = Field(description="root dir where data is stored")
    plane: str = Field(description="specify which plane to use during training")
    task: str = Field(description="specify which case will be handled during training")
    batch_size: int = Field(description="batch size during training", gt=0)
    num_workers: int = Field(description="number of workers to use during data loading, by default 1", default=6)

    @validator('plane')
    def check_if_plane_is_available(cls, value: str):
        if value not in cls._AVAILABLE_PLANES:
            raise ValueError(f"Plane must be either ['axial', 'coronal', 'sagital'] but got {value} instead")    
        return value
    
    @validator('task')
    def check_if_task_is_available(cls, value: str):
        if value not in cls._AVAILABLE_TASKS:
            raise ValueError(f"Task must be either ['abnormal', 'acl', 'meniscus'] but got {value} instead")    
        return value

class Config(BaseModel):

    """
        Config

        Base wrapper to use because it wraps every class for purpose of easy usage in project

        Args:
        :augmentation: AugmentationSettings
        :model: ModelSettings
        :train: TrainSettings
        :valid: ValidationSettings

    """

    augmentation: AugmentationSettings
    model: ModelSettings
    train: TrainSettings
    valid: ValidationSettings
