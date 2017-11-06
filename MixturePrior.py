class MixturePrior(object):
	""" Model to express the prior distribution on the Weights.
	 The prior distribution for each weight is a Mixture of 2 Gaussians (Z1 and Z2)
	 They have 0 mean and std sigma1 and sigma2, the mixing coefficient is pi.
	"""
    def __init__(self, pi, log_sigma1, log_sigma2):
		"""
		This function initializes the parameters of the Gaussian mixture
		Input parameters:
			- pi: Mixing coeffient, P(Z = Z1). Being Z1 \sim N(0,sigma1^2)
			- log_sigma1: Logarithm of the std of Z1
			- log_sigm12: Logarithm of the std of Z2
		
		Output parameters:
			- It sets the mean and sigma of the mixture as internal
			variables of the object.
		"""
        self.mean = 0      # Set the mean of both Z1 and Z2 to 0.
        
        # Computing the sigma of the mixture.
        # Is there a sigma for the mixture ?
        self.sigma_mix = pi * tf.exp(log_sigma1) +   
					    (1 - pi) * tf.exp(log_sigma2)

    def get_kl_divergence(self, gaussian1):
        # because the other compute_kl does log(sigma) and this is already set
        """
        This function computes the KL divergence between:
			- The gaussian posterior q(\theta) 
			- Our prior mixture of Gaussians.
		
		Input Parameters:
			- gaussian1: The parameters of the estimated gaussian q(\theta)
			- The parameters of the prior are stored as internal variables.
		
		Output Parameters:
			- KL(p(\theta)||q(\theta))
        """
        
        mean1, sigma1 = gaussian1  # Parameters of q(\theta)
        mean2, sigma2 = self.mean, self.sigma_mix # Parameters of p(\theta)
		
		# Compute the closed form solution of the KL divergence between two gaussians.
		# KL(p, q) = \log \frac{\sigma_2}{\sigma_1} + 
		# \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}
        kl_divergence = tf.log(sigma2) - tf.log(sigma1) + \
                        ((tf.square(sigma1) + tf.square(mean1 - mean2)) / (2 * tf.square(sigma2))) \
                        - 0.5
        return tf.reduce_mean(kl_divergence)
