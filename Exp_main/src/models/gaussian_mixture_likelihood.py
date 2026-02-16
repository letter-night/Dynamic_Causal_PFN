import torch
from torch.distributions import Normal, Categorical

class GaussianMixture:
    def __init__(self, means, stds): # input shape: n_obs x n_components
        self.n_obs = means.shape[0]
        self.num_components = means.shape[-1]
        self.components = [Normal(means[:, i], stds[:, i]) for i in range(self.num_components)]
    def sample(self, mc_samples): # n_samples = number of MC samples drawn from the mixture model
        component_idx = torch.randint(0, self.num_components,
                                      (mc_samples,))  # draw mc_samples from randomly chosen component for each obs

        return torch.stack([self.components[idx].sample() for idx in component_idx]).T # output shape: n_obs x mc_samples

    def log_prob(self, value): # input shape: either grid_shape or grid_shape x n_obs
        if len(value.shape) < self.n_obs:
            value = torch.stack([value for _ in range(self.n_obs)]).T

        component_log_probs = torch.stack([component.log_prob(value) for component in self.components])
        log_prob = torch.logsumexp(component_log_probs, dim=0) - torch.log(torch.tensor(self.num_components)).T
        return log_prob # grid_shape x n_obs
    



class GaussianMixture_pfn:
    def __init__(self, means, stds, pi): # input shape: n_obs x n_components
        """
        means, stds, pi: n_obs x n_components
        pi should sum to 1 across the last dimension.
        """
        self.n_obs = means.shape[0]
        self.num_components = means.shape[-1]
        self.pi = pi 
        self.components = [Normal(means[:, i], stds[:, i]) for i in range(self.num_components)]

        # Categorical distribution helps sample indices based on pi weights
        self.cat_dist = Categorical(probs=pi)

    def sample(self, mc_samples): 
        # Instead of randint, we use the categorical distribution defined by pi
        # component_idx shape: mc_samples x n_obs
        component_idx = self.cat_dist.sample((mc_samples,))

        # We gather the samples from the chosen components
        samples = []
        for i in range(mc_samples):
            # For each MC draw, pick the specific component for each observation
            idx = component_idx[i] # shape: (n_obs,)
            # Extract the correct normal distribution sample for each obs
            all_comp_samples = torch.stack([comp.sample() for comp in self.components], dim=1)
            chosen_samples = all_comp_samples[torch.arange(self.n_obs), idx]
            samples.append(chosen_samples)
        
        return torch.stack(samples).T # n_obs x mc_samples 


    def log_prob(self, value): # input shape: either grid_shape or grid_shape x n_obs
        if len(value.shape) == 1: # Basic broadcast handling
            value = value.unsqueeze(1)
        
        # 1. Get log probabilities from each individual component: log(p(x|k))
        # shape: n_components x value_shape x n_obs
        component_log_probs = torch.stack([component.log_prob(value) for component in self.components])

        # 2. Add the log of the mixing weights: log(pi_k) + log(p(x|k))
        # We transpose pi to align the component dimension (dim 0)
        weighted_log_probs = component_log_probs + torch.log(self.pi.T).unsqueeze(1)

        # 3. Log-Sum-Exp trick to get log(sum(pi_k * p(x|k)))
        return torch.logsumexp(weighted_log_probs, dim=0)
    