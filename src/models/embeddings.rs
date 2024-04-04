use tch::{nn, nn::Module, Device, Kind, Tensor};

/// TimestepEmbedding module for embedding timesteps.
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    /// Creates a new TimestepEmbedding module.
    ///
    /// # Arguments
    ///
    /// * `vs` - The variable store path.
    /// * `channel` - The number of input channels.
    /// * `time_embed_dim` - The dimensionality of the time embedding.
    ///
    /// # Returns
    ///
    /// A new TimestepEmbedding instance.
    pub fn new(vs: nn::Path, channel: i64, time_embed_dim: i64) -> Self {
        let linear_cfg = Default::default();
        let linear_1 = nn::linear(&vs / "linear_1", channel, time_embed_dim, linear_cfg);
        let linear_2 = nn::linear(&vs / "linear_2", time_embed_dim, time_embed_dim, linear_cfg);
        Self { linear_1, linear_2 }
    }
}

impl Module for TimestepEmbedding {
    /// Performs the forward pass of the TimestepEmbedding module.
    ///
    /// # Arguments
    ///
    /// * `xs` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the timestep embedding.
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.linear_1).silu().apply(&self.linear_2)
    }
}

/// Timesteps module for generating timestep embeddings.
#[derive(Debug)]
pub struct Timesteps {
    num_channels: i64,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    device: Device,
}

impl Timesteps {
    /// Creates a new Timesteps module.
    ///
    /// # Arguments
    ///
    /// * `num_channels` - The number of channels in the timestep embedding.
    /// * `flip_sin_to_cos` - Whether to flip sin and cos in the embedding.
    /// * `downscale_freq_shift` - The frequency shift for downscaling.
    /// * `device` - The device to use for computations.
    ///
    /// # Returns
    ///
    /// A new Timesteps instance.
    pub fn new(
        num_channels: i64,
        flip_sin_to_cos: bool,
        downscale_freq_shift: f64,
        device: Device,
    ) -> Self {
        Self { num_channels, flip_sin_to_cos, downscale_freq_shift, device }
    }
}

impl Module for Timesteps {
    /// Performs the forward pass of the Timesteps module.
    ///
    /// # Arguments
    ///
    /// * `xs` - The input tensor containing the timesteps.
    ///
    /// # Returns
    ///
    /// The output tensor representing the timestep embeddings.
    fn forward(&self, xs: &Tensor) -> Tensor {
        let half_dim = self.num_channels / 2;
        let exponent = Tensor::arange(half_dim, (Kind::Float, self.device)) * -f64::ln(10000.);
        let exponent = exponent / (half_dim as f64 - self.downscale_freq_shift);
        let emb = exponent.exp() * xs.unsqueeze(-1);
        
        let emb = if self.flip_sin_to_cos {
            Tensor::cat(&[emb.cos(), emb.sin()], -1)
        } else {
            Tensor::cat(&[emb.sin(), emb.cos()], -1)
        };
        
        if self.num_channels % 2 == 1 {
            emb.pad([0, 1, 0, 0], "constant", None)
        } else {
            emb
        }
    }
}
