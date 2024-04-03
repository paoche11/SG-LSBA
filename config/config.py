from yaml import safe_load


class Config:
    def __init__(self, config_path: str) -> None:
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))
        self.pipline_name = self.config["pipline"]
        self.device = self.config["device"]
        self.model_save_path = self.config["model_save_path"]
        self.image_save_path = self.config["image_save_path"]
        self.image_size: int = int(self.config["image_size"])
        self.image_channels: int = int(self.config["image_channels"])
        self.output_channels: int = int(self.config["output_channels"])
        self.res_layers: int = int(self.config["res_layers"])
        self.train_image_size: int = int(self.config["train_image_size"])
        self.dataset_path = self.config["dataset_path"]
        self.train_timesteps: int = int(self.config["train_timesteps"])
        self.batch_size: int = int(self.config["batch_size"])
        self.epochs: int = int(self.config["epochs"])
        self.lr: float = float(self.config["lr"])
        self.vae_save_path: str = self.config["vae_save_path"]
        self.clip_save_path: str = self.config["clip_save_path"]
        self.clip_processor_save_path: str = self.config["clip_processor_save_path"]
        self.use_text_attention: bool = bool(self.config["use_text_attention"])
        self.model_input_channels: int = int(self.config["model_input_channels"])
        self.model_output_path: str = self.config["model_output_path"]
        self.sequence_length: int = int(self.config["sequence_length"])
        self.log_path: str = self.config["log_path"]