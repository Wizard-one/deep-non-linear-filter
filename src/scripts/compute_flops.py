from torch.utils.flop_counter import FlopCounterMode
import sys
sys.path.append('.')
sys.path.append('..')
import lightning.pytorch as pl
from models.exp_jnf import JNFExp
from models.models import FTJNF
import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.


if __name__=="__main__":

    with open('config/jnf_config.yaml') as config_file: 
        config = yaml.safe_load(config_file)

    ## REPRODUCIBILITY
    pl.seed_everything(config.get('seed', 0), workers=True)
    stft_length = 512
    stft_shift = 256
    ## CONFIGURE EXPERIMENT
    model = FTJNF(**config['network'])
    exp = JNFExp(model=model,
                stft_length=stft_length,
                stft_shift=stft_shift,
                **config['experiment'])
    ### FLOPs 测试 输入为单通道数据
    x = torch.randn(1, 2, int(16000 * 4), dtype=torch.float32,device='cpu')
    X=exp.get_stft_rep(x)
    print(X)
    X=X[0]
    X = torch.concat((torch.real(X), torch.imag(X)), dim=1)
    X=X.to(device='meta')
    model=model.to(device='meta')
    model.eval()

    with FlopCounterMode(model, display=False) as fcm:
        res = model(X)[0].mean()
        flops_forward_eval = fcm.get_total_flops()
    params_eval = sum(param.numel() for param in model.parameters())
    flops_forward_eval_avg = flops_forward_eval/4e9
    print(f"flops_forward={flops_forward_eval_avg:.2f}G/s, params={params_eval/1e6:.2f} M")