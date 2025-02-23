import logging
import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from copy import copy


class DistributionWrapper:
    def __init__(self, distribution, shape, actions, action_space=None, squeeze_batch=False):
        self.distribution = distribution
        self._shape = shape
        self._actions = actions
        self.action_space = action_space
        self.squeeze_batch = squeeze_batch

    @property
    def shape(self):
        return self._shape

    def clone(self):
        cloned_distribution = copy(self.distribution)
        return DistributionWrapper(
            cloned_distribution, self._shape, self._actions,
            self.action_space, self.squeeze_batch
        )

    def cpu(self):
        if hasattr(self.distribution, "cpu"):
            try:
                new_dist = self.distribution.cpu()
            except Exception:
                new_dist = self.distribution
            return DistributionWrapper(
                new_dist, self._shape, self._actions,
                self.action_space, self.squeeze_batch
            )
        return self

    def cuda(self):
        if hasattr(self.distribution, "cuda"):
            try:
                new_dist = self.distribution.cuda()
            except Exception:
                new_dist = self.distribution
            return DistributionWrapper(
                new_dist, self._shape, self._actions,
                self.action_space, self.squeeze_batch
            )
        return self

    def numpy(self):
        # Return the stored actions as a NumPy array.
        arr = self._actions.cpu().numpy()
        # If squeeze_batch is True, assume we only want a single (unbatched) action.
        if self.squeeze_batch and arr.ndim > 0:
            return arr[0]
        return arr

    def __getattr__(self, attr):
        return getattr(self.distribution, attr)


class HybridPolicy(ActorCriticPolicy):
    """
    Custom PPO policy that expects the feature extractor to return:
      {
        "features": [B, latent_dim],
        "price_prediction": [B, 1]
      }
    It then computes the actor & critic outputs from "features" and an auxiliary
    loss from "price_prediction". The auxiliary loss (MSE vs. a target extracted from obs)
    is stored in self.aux_loss.

    The new parameter 'squeeze_actions' allows you to have the policy return a non-batched
    (i.e. 1D) action (shape (2,)) when using a non-vectorized or evaluation setting.
    """
    def __init__(self, *args, lambda_aux=0.5, squeeze_actions=False, **kwargs):
        super(HybridPolicy, self).__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux
        self.squeeze_actions = squeeze_actions  # Flag for single-env or evaluation
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"HybridPolicy initialized with lambda_aux: {self.lambda_aux}")
        self.aux_loss = torch.tensor(0.0)

    def forward(self, obs, deterministic=False):
        """
        Forward pass for sampling or determining actions given observations.
        Handles shape corrections for single vs batch environments.
        """
        feat_out = self.features_extractor(obs)
        latent_features = feat_out["features"]
        price_prediction = feat_out["price_prediction"]

        self.logger.debug(f"[Forward] obs shape: {obs.shape}")
        self.logger.debug(f"[Forward] latent_features shape: {latent_features.shape}")
        self.logger.debug(f"[Forward] price_prediction shape: {price_prediction.shape}")

        # Create the distribution and compute value
        distribution = self._get_action_dist_from_latent(latent_features)
        values = self.value_net(latent_features)
        self._last_price_prediction = price_prediction

        # Sample or choose deterministic actions
        actions = distribution.mode() if deterministic else distribution.sample()
        self.logger.debug(f"[Forward] raw actions shape: {actions.shape}")

        # If the action is 1D => shape (n_discrete,) => unsqueeze to make it (1, n_discrete)
        if actions.ndimension() == 1:
            self.logger.debug("[Forward] Detected 1D action; unsqueezing to add batch dim.")
            actions = actions.unsqueeze(0)

        # Squeeze if single-batch environment and user requested it
        if self.squeeze_actions and actions.size(0) == 1:
            self.logger.debug("[Forward] squeeze_actions=True, squeezing out batch dimension.")
            actions = actions.squeeze(0)  # (n_discrete,)

        self.logger.debug(f"[Forward] final actions shape: {actions.shape}")

        # Wrap the distribution, passing the actions and the action space
        if isinstance(actions, torch.Tensor):
            new_shape = actions.shape
        else:
            new_shape = (len(actions),)
        wrapped_dist = DistributionWrapper(distribution, new_shape, actions,
                                           self.action_space, squeeze_batch=self.squeeze_actions)
        return actions, values, wrapped_dist

    def evaluate_actions(self, obs, actions):
        """
        Evaluates actions for the given observations. Handles potential shape corrections
        for multi-discrete actions (e.g., from (2,) => (1, 2)).
        """
        feat_out = self.features_extractor(obs)
        latent_features = feat_out["features"]
        price_prediction = feat_out["price_prediction"]

        distribution = self._get_action_dist_from_latent(latent_features)
        values = self.value_net(latent_features)

        # If receiving a single env action of shape (2,), expand to (1,2) for log_prob
        if actions.ndimension() == 1 and len(actions) == 2:
            self.logger.debug("[evaluate_actions] Expanding action from shape (2,) to (1,2).")
            actions = actions.unsqueeze(0)

        self.logger.debug(f"[evaluate_actions] actions shape used for log_prob: {actions.shape}")
        log_prob = distribution.log_prob(actions)

        self.logger.debug(f"[evaluate_actions] Raw log_prob type: {type(log_prob)}; shape: {getattr(log_prob, 'shape', 'N/A')}")
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=values.device)
        self.logger.debug(f"[evaluate_actions] log_prob shape before summing: {log_prob.shape}")

        # Sum across discrete dimensions for multi-discrete distributions
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=1)
        self.logger.debug(f"[evaluate_actions] log_prob shape after summing: {log_prob.shape}")

        # Compute distribution entropy
        entropy = distribution.entropy()

        # Compute auxiliary loss for price prediction
        B = obs.shape[0]
        window_size = self.features_extractor.window_size
        total_features = self.features_extractor.input_dim

        # Reshape obs to retrieve target price
        obs_reshaped = obs[:, :window_size * total_features].view(B, window_size, total_features)
        self.logger.debug(f"[evaluate_actions] Reshaped obs shape: {obs_reshaped.shape}")
        sample_last_timestep = obs_reshaped[0, -1, :].detach().cpu().numpy()
        self.logger.debug(f"[evaluate_actions] Sample last timestep values: {sample_last_timestep}")

        target_price = obs_reshaped[:, -1, 0].float().to(values.device).squeeze(-1)
        self.logger.debug(f"[evaluate_actions] Extracted target_price shape: {target_price.shape}")
        self.logger.debug(f"[evaluate_actions] Extracted target_price values (first 5): {target_price[:5]}")

        price_prediction = price_prediction.squeeze(-1)
        self.logger.debug(f"[evaluate_actions] price_prediction shape after squeeze: {price_prediction.shape}")
        self.logger.debug(f"[evaluate_actions] price_prediction values (first 5): {price_prediction[:5]}")

        # Adjust shape if mismatch
        if target_price.shape != price_prediction.shape:
            self.logger.debug(
                f"[evaluate_actions] Shape mismatch: target_price {target_price.shape} vs price_prediction {price_prediction.shape}. Reshaping target_price."
            )
            target_price = target_price.view_as(price_prediction)
        
        aux_loss = F.mse_loss(price_prediction, target_price)
        self.aux_loss = aux_loss
        self.logger.debug(f"[evaluate_actions] Final shapes -- target_price: {target_price.shape}, "
                          f"price_prediction: {price_prediction.shape}, aux_loss: {aux_loss.item()}")

        return values, log_prob, entropy, aux_loss

    def _get_action_dist_from_latent(self, latent):
        """
        Produces the distribution from the latent features (e.g., multi-discrete for action space (3, 10)).
        """
        latent_pi, _ = self.mlp_extractor(latent)
        action_logits = self.action_net(latent_pi)
        self.logger.debug(f"[ActionDist] action_logits shape: {action_logits.shape}")
        dist = self.action_dist.proba_distribution(action_logits)
        self.logger.debug(f"[ActionDist] Distribution type: {type(dist)}")
        return dist

    def compute_loss(self, advantages, log_prob, value_preds, values, entropy):
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=values.device)

        logging.debug(f"[compute_loss] log_prob shape before sum: {log_prob.shape}")
        if log_prob.ndim > 1:
            log_prob = torch.sum(log_prob, dim=1)
        logging.debug(f"[compute_loss] log_prob shape after sum: {log_prob.shape}")
        
        policy_loss = -(log_prob * advantages).mean()
        value_loss = F.mse_loss(values, value_preds)
        ppo_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
        
        # Add auxiliary loss (price prediction)
        total_loss = ppo_loss + self.lambda_aux * self.aux_loss
        return total_loss
