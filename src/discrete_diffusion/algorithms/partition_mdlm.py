"""PartitionMDLM algorithm definition extracted from :mod:`algorithms.algo`."""

import torch

from .. import utils
from .mdlm import MDLM


class PartitionMDLM(MDLM):
  def __init__(self, config, tokenizer):
    self.sampling_mode = config.algo.sampling_mode
    self.post_process_fp64 = getattr(config.algo, 'post_process_fp64', False)
    super().__init__(config, tokenizer)

  def _validate_configuration(self):
    assert not self.time_conditioning, \
      "Partition MDLM cannot be time conditioned."

    assert self.sampling_mode in {'naive', 'efficient-uniform',
                'efficient-non-uniform'}, self.sampling_mode
    return super()._validate_configuration()
  
  def _q_xt_partition(self, x, alpha_t):
    #  Probability of being in group 0 is alpha_t.
    #  -> masking probability is 1 - alpha_t
    group_idxs = torch.rand(
        *x.shape, device=x.device) < 1 - alpha_t
    return group_idxs.to(int)

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    del xt
    if self.post_process_fp64:
      model_output = model_output.to(torch.float64)
    model_output[:, :, self.mask_id] = self.neg_infinity
    return model_output

  def forward(self, xt, sigma, group_idxs=None, 
              clean_positions=None, noisy_positions=None, 
              concrete_lengths=None, use_inference_mode=False):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      model_output = self.backbone(xt, 
        sigma=sigma, 
        group_idxs=group_idxs,
        clean_positions=clean_positions,
        noisy_positions=noisy_positions,
        concrete_lengths=concrete_lengths,
        use_inference_mode=use_inference_mode)
      if use_inference_mode and self.config.sampling.p_nucleus < 1:
        model_output = utils.top_k_top_p_filtering(
          model_output, top_p=self.config.sampling.p_nucleus)
    return self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)

  def nll(self, x0,
          current_accumulation_step=None, train_mode=False):
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    t_complement = 1 - t

    alpha_t = self.noise.alpha_t(t)
    dalpha_t = self.noise.alpha_prime_t(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    alpha_t_complement = self.noise.alpha_t(t_complement)
    dalpha_t_complement = self.noise.alpha_prime_t(t_complement)
    alpha_t_complement = alpha_t_complement.unsqueeze(-1)
    dalpha_t_complement = dalpha_t_complement.unsqueeze(-1)
    assert alpha_t_complement.ndim == 2

    group_idxs = self._q_xt_partition(x0, alpha_t)
    log_x_theta = self.forward(x0, torch.zeros_like(sigma),
                               group_idxs)
    # For the group 0, the group 1 represents mask tokens.
    #  Hence, loss for tokens in the group 1 should be scaled 
    #  using alpha_t and dalpha_t.
    nll_alpha_t = torch.where(group_idxs != 0, alpha_t, 
                              alpha_t_complement)
    nll_dalpha_t = torch.where(group_idxs != 0, dalpha_t, 
                               dalpha_t_complement)
    tokens_nll = self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=None,  # is ignored in MDLM
      x0=x0,
      alpha_t=nll_alpha_t,
      dalpha_t=nll_dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

    if train_mode:
      return tokens_nll / 2
    else:
      # As if group 1 represents mask tokens
      return torch.where(group_idxs == 1, tokens_nll, 0.)


