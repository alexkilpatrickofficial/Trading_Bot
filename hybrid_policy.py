import logging
import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from copy import copy

# DistributionWrapper to pass along actions and distribution information.
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
        return DistributionWrapper(cloned_distribution, self._shape, self._actions,
                                   self.action_space, self.squeeze_batch)

    def cpu(self):
        if hasattr(self.distribution, "cpu"):
            try:
                new_dist = self.distribution.cpu()
            except Exception:
                new_dist = self.distribution
            return DistributionWrapper(new_dist, self._shape, self._actions,
                                       self.action_space, self.squeeze_batch)
        return self

    def cuda(self):
        if hasattr(self.distribution, "cuda"):
            try:
                new_dist = self.distribution.cuda()
            except Exception:
                new_dist = self.distribution
            return DistributionWrapper(new_dist, self._shape, self._actions,
                                       self.action_space, self.squeeze_batch)
        return self

    def numpy(self):
        arr = self._actions.cpu().numpy()
        if self.squeeze_batch and arr.ndim > 0:
            return arr[0]
        return arr

    def __getattr__(self, attr):
        return getattr(self.distribution, attr)


class HybridPolicy(ActorCriticPolicy):
    """
    Custom PPO policy that uses a feature extractor which returns:
      {
        "features": [B, latent_dim],
        "price_prediction": [B, 1]
      }
    The policy computes actor and critic outputs from "features" and incorporates
    an auxiliary loss (MSE vs. a target extracted from obs) from "price_prediction".
    This auxiliary loss is stored in self.aux_loss and added (weighted by lambda_aux)
    to the overall loss.
    
    The parameter 'squeeze_actions' optionally removes the batch dimension when using a single environment.
    """
    def __init__(self, *args, lambda_aux=0.5, squeeze_actions=False, **kwargs):
        super(HybridPolicy, self).__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux
        self.squeeze_actions = squeeze_actions
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"HybridPolicy initialized with lambda_aux: {self.lambda_aux}")
        self.aux_loss = torch.tensor(0.0)

        # Initialize loss metrics attributes for logging
        self.policy_loss = None
        self.value_loss = None
        self.total_loss = None
        self.explained_variance = None

    def forward(self, obs, deterministic=False):
        """
        Forward pass for sampling or selecting actions.
        Processes observations through the feature extractor,
        then computes latent features and price prediction.
        Creates a categorical distribution (Discrete action space) from latent features.
        """
        feat_out = self.features_extractor(obs)
        latent_features = feat_out["features"]
        price_prediction = feat_out["price_prediction"]

        self.logger.debug(f"[Forward] obs shape: {obs.shape}")
        self.logger.debug(f"[Forward] latent_features shape: {latent_features.shape}")
        self.logger.debug(f"[Forward] price_prediction shape: {price_prediction.shape}")

        distribution = self._get_action_dist_from_latent(latent_features)
        values = self.value_net(latent_features)
        self._last_price_prediction = price_prediction

        actions = distribution.mode() if deterministic else distribution.sample()
        self.logger.debug(f"[Forward] raw actions shape: {actions.shape}")

        # Ensure actions are 2D (batch, action_dim)
        if actions.ndimension() == 1:
            self.logger.debug("[Forward] Action is 1D; unsqueezing to add batch dimension.")
            actions = actions.unsqueeze(1)
        
        if self.squeeze_actions and actions.size(0) == 1:
            self.logger.debug("[Forward] squeeze_actions=True; squeezing batch dimension.")
            actions = actions.squeeze(0)  # now (action_dim,)

        self.logger.debug(f"[Forward] final actions shape: {actions.shape}")

        wrapped_dist = DistributionWrapper(distribution, actions.shape, actions,
                                           self.action_space, squeeze_batch=self.squeeze_actions)
        return actions, values, wrapped_dist

    def evaluate_actions(self, obs, actions):
        """
        Evaluate given actions for the provided observations.
        Expands action shape if necessary and computes the auxiliary loss based on price prediction.
        """
        feat_out = self.features_extractor(obs)
        latent_features = feat_out["features"]
        price_prediction = feat_out["price_prediction"]

        distribution = self._get_action_dist_from_latent(latent_features)
        values = self.value_net(latent_features)

        if actions.ndimension() == 1:
            self.logger.debug("[evaluate_actions] Expanding action from shape (action_dim,) to (1, action_dim).")
            actions = actions.unsqueeze(0)

        self.logger.debug(f"[evaluate_actions] actions shape used for log_prob: {actions.shape}")
        log_prob = distribution.log_prob(actions)
        self.logger.debug(f"[evaluate_actions] Raw log_prob type: {type(log_prob)}; shape: {getattr(log_prob, 'shape', 'N/A')}")
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=values.device)
        self.logger.debug(f"[evaluate_actions] log_prob shape before processing: {log_prob.shape}")
        # For discrete distribution, log_prob is already 1D per sample.
        self.logger.debug(f"[evaluate_actions] log_prob shape after processing: {log_prob.shape}")

        entropy = distribution.entropy()

        # Auxiliary loss computation for price prediction:
        B = obs.shape[0]
        window_size = self.features_extractor.window_size
        total_features = self.features_extractor.input_dim

        obs_reshaped = obs[:, :window_size * total_features].view(B, window_size, total_features)
        self.logger.debug(f"[evaluate_actions] Reshaped obs shape: {obs_reshaped.shape}")
        target_price = obs_reshaped[:, -1, 0].float().to(values.device).squeeze(-1)
        self.logger.debug(f"[evaluate_actions] target_price shape: {target_price.shape}")
        self.logger.debug(f"[evaluate_actions] target_price (first 5): {target_price[:5]}")

        price_prediction = price_prediction.squeeze(-1)
        self.logger.debug(f"[evaluate_actions] price_prediction shape after squeeze: {price_prediction.shape}")
        self.logger.debug(f"[evaluate_actions] price_prediction (first 5): {price_prediction[:5]}")

        if target_price.shape != price_prediction.shape:
            self.logger.debug(f"[evaluate_actions] Shape mismatch: target_price {target_price.shape} vs price_prediction {price_prediction.shape}. Reshaping target_price.")
            target_price = target_price.view_as(price_prediction)
        
        aux_loss = F.mse_loss(price_prediction, target_price)
        self.aux_loss = aux_loss
        self.logger.debug(f"[evaluate_actions] Final aux_loss: {aux_loss.item()}")

        return values, log_prob, entropy, aux_loss

    def _get_action_dist_from_latent(self, latent):
        """
        Produces the action distribution from the latent features.
        For a Discrete action space, the network output (logits) should be of shape (batch, num_actions).
        """
        latent_pi, _ = self.mlp_extractor(latent)
        action_logits = self.action_net(latent_pi)
        self.logger.debug(f"[ActionDist] action_logits shape: {action_logits.shape}")
        dist = self.action_dist.proba_distribution(action_logits)
        self.logger.debug(f"[ActionDist] Distribution type: {type(dist)}")
        return dist

    def compute_loss(self, advantages, log_prob, value_preds, values, entropy):
        """
        Computes the PPO loss with an added auxiliary MSE loss for price prediction.
        Stores individual components (policy_loss, value_loss, total_loss) for logging.
        """
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32, device=values.device)
        self.logger.debug(f"[compute_loss] log_prob shape before sum: {log_prob.shape}")

        # Compute policy loss and value loss
        policy_loss = -(log_prob * advantages).mean()
        value_loss = F.mse_loss(values, value_preds)
        ppo_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        total_loss = ppo_loss + self.lambda_aux * self.aux_loss

        # Store losses for external logging (e.g., via a custom callback)
        self.policy_loss = policy_loss.item()
        self.value_loss = value_loss.item()
        self.total_loss = total_loss.item()
        # If the model computes explained variance, store it as well.
        if hasattr(self, "explained_variance"):
            self.explained_variance = self.explained_variance
        else:
            self.explained_variance = None

        self.logger.debug(f"[compute_loss] Policy loss: {self.policy_loss}, Value loss: {self.value_loss}, PPO loss: {ppo_loss.item()}, Aux loss: {self.aux_loss.item()}, Total loss: {self.total_loss}")
        return total_loss
