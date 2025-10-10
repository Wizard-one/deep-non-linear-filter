import json
from typing import Any, Literal
import torch
from torch import nn
from models.exp_enhancement import EnhancementExp
from models.models import FTJNF
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore as DNSMOS
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
import os
import numpy as np
from dcnet.models.utils.general_steps import test_step_write_example

class JNFExp(EnhancementExp):

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float,
                 weight_decay: float, 
                 loss_alpha: float,
                 stft_length: int,
                 stft_shift: int,
                 cirm_comp_K: float,
                 cirm_comp_C: float, 
                 reference_channel: int = 0):
        super(JNFExp, self).__init__(model=model, cirm_comp_K=cirm_comp_K, cirm_comp_C=cirm_comp_C)

        self.model = model

        self.stft_length = stft_length
        self.stft_shift = stft_shift

        self.cirm_K = cirm_comp_K
        self.cirm_C = cirm_comp_C

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_alpha = loss_alpha

        self.reference_channel = reference_channel
        self.dnsmos = DNSMOS(fs=16000, personalized=False)
        self.input_dnsmos = DNSMOS(fs=16000, personalized=False)
        self.input_pesq_wb = PESQ(fs=16000, mode="wb")
        self.pesq_wb = PESQ(fs=16000, mode="wb")


        #self.example_input_array = torch.from_numpy(np.ones((2, 6, 513, 75), dtype=np.float32))

    def forward(self, input):
        speech_mask = self.model(input)
        return speech_mask
    
    @classmethod
    def get_init_params(cls, config):
        init_params = {
            "model": FTJNF(**config['network']),
            "learning_rate": config['experiment']['learning_rate'],
            "weight_decay": config['experiment']['weight_decay'],
            "loss_alpha": config['experiment']['loss_alpha'],
            "stft_length": config['data']['stft_length_samples'],
            "stft_shift": config['data']['stft_shift_samples'],
            "cirm_comp_K": config['experiment']['cirm_comp_K'],
            "cirm_comp_C": config['experiment']['cirm_comp_C'],
            # "reference_channel": config['experiment'].get('reference_channel', 0)
        }
        return init_params  


    @staticmethod
    def get_angle_mask(paras, batch_idx):
        # TODO: 非常不好
        result, is_zero_target, is_zero_inter = {}, [], []
        if not isinstance(paras, list):
            raise NotImplementedError("paras should be a list of dicts")
        for p in paras:
            if "angle" not in result:
                result["angle"] = []
            if "angle_inter" not in result:
                result["angle_inter"] = []
            if "sample_id" not in result:
                result["sample_id"] = []
            if "batch_id" not in result:
                result["batch_id"] = []
            result["angle"].append(p["angle"])
            result["angle_inter"].append(p["angle_inter"])
            result["sample_id"].append(p["index"])
            result["batch_id"].append(batch_idx)
            is_zero_target.append(p["is_zero_target"])
            is_zero_inter.append(p["is_zero_interfer"])
        for key in result:
            result[key] = np.array(result[key])
        # TODO 是否转到torch.tensor device 上会更快点?
        return (
            result,
            np.array(is_zero_target),
            np.array(is_zero_inter),
        )    

    def on_test_epoch_start(self):
        if self.trainer.log_dir is None:
            raise ValueError("Logger log_dir is not set. Cannot save metrics.")
        os.makedirs(self.trainer.log_dir, exist_ok=True)
        return super().on_test_epoch_start()
    
    def target_metric(self, x_ref, yr_hat, yr):
        self.input_dnsmos.update(x_ref)
        self.dnsmos.update(yr_hat)
        self.input_pesq_wb.update(x_ref, yr)
        self.pesq_wb.update(yr_hat, yr)

    def test_step(self, batch, batch_idx):
        x, ys, paras = batch
        B = ys.shape[0]
        yr = ys.squeeze(1)
        meta, zero_target_mask, zero_inter_mask = self.get_angle_mask(paras, batch_idx)

        yr_hat, Yr_hat, out_mask = self.forward(x)
        wavname = os.path.basename(f"{paras[0]['index']}_{paras[0]['angle']:.2f}.wav")
        if ((~zero_target_mask).sum()) > 0:
            self.target_metric(
                x_ref=x[~zero_target_mask, 0],
                yr_hat=yr_hat[~zero_target_mask],
                yr=yr[~zero_target_mask],
            )
        if paras[0]["index"] < 10:
            if self.name != "notag" and self.trainer.log_dir is not None:
                rootfolder = self.trainer.log_dir.split("/")[0]
                folder = f"{rootfolder}/{self.save_to}/{self.name}"
                os.makedirs(folder, exist_ok=True)
            else:
                folder = self.trainer.log_dir
            test_step_write_example(
                self=self,
                xr=x / torch.max(torch.abs(x)),
                yr=ys,
                yr_hat=yr_hat.unsqueeze(0),
                sample_rate=16000,
                paras=paras,
                result_dict={},
                wavname=wavname,
                exp_save_path=folder,
            )
    def on_test_epoch_end(self) -> None:
        #     """calculate heavy metrics for every N epochs"""
        if self.name != "notag" and self.trainer.log_dir is not None:
            rootfolder = self.trainer.log_dir.split("/")[0]
            folder = f"{rootfolder}/{self.save_to}"
            os.makedirs(folder, exist_ok=True)
        elif self.trainer.log_dir is not None:
            folder = self.trainer.log_dir
        else:
            folder = "."
        dnsmos_input = self.input_dnsmos.compute()
        dnsmos = self.dnsmos.compute()
        pesq_input = self.input_pesq_wb.compute()
        pesq = self.pesq_wb.compute()
        if self.trainer.is_global_zero:
            json_path = os.path.join(folder, "loss_result.json")
            result = {}
            result.update(
                {
                    "val/input_dnsmos_p808": dnsmos_input[0].item(),
                    "val/input_dnsmos_sig": dnsmos_input[1].item(),
                    "val/input_dnsmos_bak": dnsmos_input[2].item(),
                    "val/input_dnsmos_ovr": dnsmos_input[3].item(),
                    "val/dnsmos_p808": dnsmos[0].item(),
                    "val/dnsmos_sig": dnsmos[1].item(),
                    "val/dnsmos_bak": dnsmos[2].item(),
                    "val/dnsmos_ovr": dnsmos[3].item(),
                    "val/input_pesq_wb": pesq_input.item(),
                    "val/pesq_wb": pesq.item(),
                }
            )
            with open(json_path, "w") as f:
                json.dump({k: float(v) for k, v in result.items()}, f, indent=2)
        self.input_dnsmos.reset()
        self.dnsmos.reset()
        self.input_pesq_wb.reset()
        self.pesq_wb.reset()