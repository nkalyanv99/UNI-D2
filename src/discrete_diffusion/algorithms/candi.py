""""CANDI implementation from https://arxiv.org/abs/2510.22510"""

import torch
import torch.nn.functional as F
from discrete_diffusion.algorithms.base import Diffusion
import omegaconf
import hydra

class CANDI(Diffusion):
    def __init__(self, config, tokenizer):
        vocab_size = len(tokenizer) + 1 # accounting for mask token
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        # getting custom forward process
        fp_cfg = getattr(self.config.algo, "forward_process", None)
        fp_config = omegaconf.OmegaConf.create(fp_cfg)
        self._forward_process = hydra.utils.instantiate(
            fp_config, tokenizer=self.tokenizer, schedule=self.noise, _recursive_=False
        )
        self.temp = config.algo.get("temp", 1.0)

    def _validate_configuration(self):
        # for further extension on CANDI variants, ensure that names always include "candi"

        # Validate that forward process is compatible with CANDI
        assert "candi" in self._forward_process.name.lower()

        # Validate that the backbone is also compatible
        assert "candi" in self.config.model.name 
        return super()._validate_configuration()
    
    def _process_model_output(self, xt, model_output, reveal_mask): 
        if xt.ndim == 2: 
            xt_tokens = xt
        else: 
            xt_tokens = xt.argmax(dim=-1)

        # if not training, apply temperature 
        if not self.training:
            model_output = model_output / self.temp
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        reveal_mask = reveal_mask.bool()
        model_output[reveal_mask] = self.neg_infinity
        model_output[reveal_mask, xt_tokens[reveal_mask]] = 0
        return model_output
    
    def nll_per_token(self, log_x_theta, alpha_t, dalpha_t, x0_tokens, **kwargs): 
        """Computes the negative log-likelihood per token as in Equation 13 of CANDI paper."""

        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0_tokens[:, :, None]
        ).squeeze(-1)
        nll = log_p_theta * dalpha_t / (1 - alpha_t)
        return nll
    
    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        del output_tokens  # Unused

        t = self._sample_t(x0.shape[0], current_accumulation_step)
        alpha_t = self.noise.alpha_t(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        
        
        assert t.shape[0] == x0.shape[0]

        noisy_input = self._forward_process.forward(x0, t)
        log_x_theta = self.forward(**noisy_input)
        return self.nll_per_token(
                log_x_theta=log_x_theta, 
                x0_tokens=x0,
                **noisy_input,
           ) 

    def forward(self, **kwargs): 
        model_output = self.backbone(**kwargs)
        return self._process_model_output(model_output=model_output, xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask'])

    def prior_sample(self, *batch_dims): 
        sigma = self.noise.sigma_t(torch.tensor(.999).to(self.device))
        noise = torch.randn(
            *batch_dims, self.vocab_size-1, dtype=torch.float32, device=self.device
        )  * sigma
        return noise