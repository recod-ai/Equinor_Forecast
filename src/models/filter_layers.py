import tensorflow as tf


class WaveletDenoiseLayer(tf.keras.layers.Layer):
    """
    Single-level Haar wavelet transform + soft-threshold + inverse transform.
    For simplicity, this layer:
      - Expects input shape (batch, horizon).
      - Assumes 'horizon' is even.
      - Uses a fixed or trainable threshold.
    """
    def __init__(self, threshold=0.1, trainable_threshold=False, **kwargs):
        """
        Args:
            threshold (float): initial threshold for soft thresholding of detail coefficients.
            trainable_threshold (bool): if True, threshold becomes a trainable parameter.
        """
        super(WaveletDenoiseLayer, self).__init__(**kwargs)
        self.threshold_value = threshold
        self.trainable_threshold = trainable_threshold

    def build(self, input_shape):
        # input_shape: (batch, horizon)
        # We'll store 'threshold' as a tf.Variable if trainable, else a constant.
        if self.trainable_threshold:
            # Make threshold a trainable parameter
            self.threshold = self.add_weight(
                name="wavelet_threshold",
                shape=(),
                initializer=tf.constant_initializer(self.threshold_value),
                trainable=True
            )
        else:
            # Just store a constant
            self.threshold = tf.constant(self.threshold_value, dtype=self.dtype)
        super(WaveletDenoiseLayer, self).build(input_shape)

    def call(self, inputs):
        """
        inputs: (batch_size, horizon).
        """
        # Ensure horizon is even
        horizon = tf.shape(inputs)[1]
        # Split even, odd
        x_even = inputs[:, ::2]  # (batch, horizon/2)
        x_odd  = inputs[:, 1::2] # (batch, horizon/2)

        # Approx + Detail
        approx = (x_even + x_odd) / tf.sqrt(tf.constant(2.0, dtype=self.dtype))
        detail = (x_even - x_odd) / tf.sqrt(tf.constant(2.0, dtype=self.dtype))

        # Soft threshold on detail
        thr = tf.abs(detail) - self.threshold
        # ReLU part: max(0, |detail|-threshold)
        thr = tf.nn.relu(thr)  # (batch, horizon/2)
        # Recover sign
        detail_denoised = tf.sign(detail) * thr

        # Inverse Haar Transform
        x_even_rec = (approx + detail_denoised) / tf.sqrt(tf.constant(2.0, dtype=self.dtype))
        x_odd_rec  = (approx - detail_denoised) / tf.sqrt(tf.constant(2.0, dtype=self.dtype))

        # Interleave even, odd to recover shape (batch, horizon)
        # We'll stack [x_even_rec, x_odd_rec] along the last dimension => (batch, horizon/2, 2)
        # then reshape to (batch, horizon)
        reconstructed = tf.stack([x_even_rec, x_odd_rec], axis=2)  # (batch, horizon/2, 2)
        output = tf.reshape(reconstructed, tf.shape(inputs))       # (batch, horizon)
        return output

    def get_config(self):
        config = super(WaveletDenoiseLayer, self).get_config()
        config.update({
            "threshold_value": self.threshold_value,
            "trainable_threshold": self.trainable_threshold,
        })
        return config
    

import tensorflow as tf
from tensorflow.keras.layers import Layer

class PolynomialSmoothingLayer(Layer):
    """
    Fits a single polynomial of user-defined degree across the entire batch.
    The polynomial is used to generate a smoothed version of the input.
    """
    def __init__(self, degree=3, **kwargs):
        """
        Args:
            degree (int): polynomial degree
        """
        super(PolynomialSmoothingLayer, self).__init__(**kwargs)
        self.degree = degree

    def build(self, input_shape):
        # input_shape: (batch, horizon)
        self.horizon = input_shape[-1]

        # We have (degree+1) coefficients: c_0, c_1, ..., c_degree
        # These will be trainable scalars.
        self.coeffs = self.add_weight(
            name='poly_coeffs',
            shape=(self.degree + 1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        super(PolynomialSmoothingLayer, self).build(input_shape)

    def call(self, inputs):
        """
        inputs shape: (batch, horizon)
        We'll:
            1) Build a polynomial basis matrix [1, t, t^2, ..., t^degree]
            2) Multiply by the learned coefficients.
            3) That yields a single 'smoothed' polynomial curve. 
            4) Optionally, we can combine (e.g. add) that curve with the original input or just return the curve.
        """
        batch_size = tf.shape(inputs)[0]
        horizon = tf.cast(self.horizon, tf.float32)

        # We'll define a range t=0..(horizon-1)
        t = tf.range(start=0, limit=self.horizon, dtype=tf.float32)  # (horizon,)

        # Create the design matrix for the polynomial basis => shape (horizon, degree+1)
        # E.g. for degree=3 => [1, t, t^2, t^3]
        basis = []
        for d in range(self.degree + 1):
            basis.append(tf.pow(t, d))
        # stack => shape (degree+1, horizon)
        basis = tf.stack(basis, axis=0)
        # transpose => shape (horizon, degree+1)
        basis = tf.transpose(basis)  # shape: (horizon, degree+1)

        # Multiply basis by our trainable coefficients => shape (horizon,)
        # poly_curve[t] = sum_{d=0..degree} (coeffs[d] * t^d)
        poly_curve = tf.matmul(basis, tf.expand_dims(self.coeffs, axis=-1))  # (horizon, 1)
        poly_curve = tf.squeeze(poly_curve, axis=-1)  # (horizon,)

        # We want to broadcast this same polynomial curve across the batch
        # => shape (batch, horizon)
        poly_curve_batched = tf.tile(tf.expand_dims(poly_curve, axis=0), [batch_size, 1])

        return poly_curve_batched

    def get_config(self):
        config = super(PolynomialSmoothingLayer, self).get_config()
        config.update({
            "degree": self.degree,
        })
        return config

    
'''

\section*{Wavelet Denoising Enhances Physics-Informed Forecasting in Hybrid Models}

\subsection*{Overview}

In hybrid time series forecasting models that integrate both data-driven trends and physics-informed layers, the relative contribution of each component reflects its ability to capture meaningful signal from noisy observational data. In our application to oil and gas production forecasting, we observed that introducing a \texttt{WaveletDenoiseLayer} significantly increased the relative contribution of the physics-based forecast. This section provides a theoretical explanation grounded in signal processing, time series theory, and physics-informed machine learning.

\subsection*{Theoretical Motivation}

Let $y_{\text{trend}}$ and $y_{\text{phys}}$ represent forecasts from the trend-based and physics-informed branches, respectively. The fused prediction $y_{\text{fusion}}$ is computed via concatenation followed by a dense layer:
\[
y_{\text{fusion}} = f_{\text{dense}}([y_{\text{trend}}, y_{\text{phys}}]),
\]
where $f_{\text{dense}}$ is trained to optimize predictive performance.

The physics-informed forecast $y_{\text{phys}}$ is derived from Darcy-based equations, which, while physically grounded, are more sensitive to measurement noise in features such as pressure and porosity. This makes $y_{\text{phys}}$ inherently more volatile and susceptible to overfitting to local irregularities.

\subsection*{Wavelet Denoising as a Signal Refinement Operator}

Wavelet transforms are a powerful tool in signal processing for separating structure from noise. A single-level Haar transform decomposes a signal $x(t)$ into approximation ($A$) and detail ($D$) coefficients:
\[
A = \frac{x_{\text{even}} + x_{\text{odd}}}{\sqrt{2}}, \quad D = \frac{x_{\text{even}} - x_{\text{odd}}}{\sqrt{2}}.
\]
Applying soft thresholding to the detail coefficients attenuates high-frequency noise while preserving underlying trends. The denoised signal $\tilde{x}(t)$ is reconstructed from the filtered coefficients, effectively suppressing transient artifacts.

When applied to the fused forecast $y_{\text{fusion}}$, the \texttt{WaveletDenoiseLayer} performs the operation:
\[
\tilde{y}_{\text{fusion}} = \mathcal{W}^{-1} \circ \mathcal{T}_{\theta} \circ \mathcal{W}(y_{\text{fusion}}),
\]
where $\mathcal{W}$ is the wavelet transform, $\mathcal{T}_{\theta}$ is soft-thresholding with trainable threshold $\theta$, and $\mathcal{W}^{-1}$ is the inverse transform.

\subsection*{Why Physics-Based Forecasts Benefit More}

We hypothesize the following mechanisms explain the increased utility of the physics-based layer after wavelet denoising:

\begin{enumerate}
    \item \textbf{Noise Disparity Hypothesis:} The physics-informed forecast $y_{\text{phys}}$ contains higher-frequency noise due to real-world measurement imperfections. In contrast, $y_{\text{trend}}$ is already smooth due to the polynomial trend fitting. Denoising disproportionately benefits $y_{\text{phys}}$ by removing its noise and making it more competitive in fusion.
    
    \item \textbf{Signal Retention Principle:} Wavelet denoising retains low-frequency, physically meaningful components (e.g., depletion curves, pressure decay) that are central to Darcy-based models. This enhances the interpretability and utility of the physical forecast without requiring it to compete with data-driven trends on noisy grounds.
    
    \item \textbf{Gradient Attribution Effect:} With reduced noise in $y_{\text{phys}}$, gradients from the loss function propagate more coherently through the physics branch during training. This improves the optimization dynamics and allows the fusion layer to assign greater weight to physical signals.
    
    \item \textbf{Explainability-Alignment Advantage:} In oil and gas, stakeholders favor models that align with domain knowledge. A denoised $y_{\text{phys}}$ better reflects interpretable physical phenomena, thereby encouraging models to rely more heavily on it during learning and inference.
\end{enumerate}


By suppressing high-frequency noise, the \texttt{WaveletDenoiseLayer} enhances the relative clarity and stability of the physics-informed forecast. This shift allows the hybrid model to assign greater predictive weight to the physically grounded component, improving both performance and interpretability. This finding supports the integration of signal-processing priors into hybrid physical-ML pipelines, especially in domains where trust and explainability are essential.

'''

#--------------------------------------------------------------------------------------------------------------------------------

'''

\section{Theoretical Discussion on Enhanced Physics Contribution via Wavelet Denoising}

The observed increase in the physics-informed layer's contribution following the application of the \textit{WaveletDenoiseLayer} can be explained through a combination of principles from signal processing, physics-informed learning, and domain-specific considerations in oil and gas production forecasting.

The forecasting model is fundamentally hybrid, composed of a polynomial-based \textbf{trend layer} and a \textbf{physics-informed (PIN) layer} employing Darcy-type flow equations. Although the polynomial layer is adept at capturing smooth, global trends inherent in production decline curves, its flexibility makes it prone to overfitting high-frequency noise. Conversely, the PIN layer, grounded explicitly in physical laws, is sensitive to noisy input measurements from pressure gauges and flow sensors, which frequently contain instrumentation or environmental noise.

Introducing the wavelet denoising step after fusion effectively addresses these limitations. The theoretical basis for this improvement is multifaceted:

\begin{itemize}
    \item \textbf{Signal Frequency Decomposition via Wavelets}: 
    Wavelet transforms perform localized decomposition of time series into frequency bands. By employing a Haar wavelet and subsequent thresholding, the denoising layer removes primarily high-frequency noise while preserving underlying signals that are physically meaningful, such as reservoir pressure trends or smooth production declines. In contrast to polynomial fitting, wavelets preserve local discontinuities critical in transient flow periods typical in oil and gas reservoirs.

    \item \textbf{Differential Impact of Noise on PIN and Trend Layers}:
    The PIN layer, constructed upon Darcy equations, assumes smooth underlying physical processes—flow through porous media governed by continuous pressure gradients. Noise, especially high-frequency oscillations, adversely impacts the integrity of these physical assumptions, diluting the predictive power of physics-based constraints. By selectively reducing high-frequency noise, wavelet denoising significantly enhances alignment between measured data and physical theory, allowing the physics-informed layer to contribute more effectively to the final forecast.

    \item \textbf{Learning Dynamics during Model Training}:
    The \textit{WaveletDenoiseLayer}, utilizing a trainable threshold, adapts its denoising intensity during training. This adaptation implicitly learns the optimal noise-frequency boundary separating physically interpretable signals from spurious variations. Hence, the layer dynamically enhances the PIN layer's robustness and predictive capability, amplifying its relative contribution during the fusion process.
\end{itemize}

Formally, consider the hybrid forecast model output as the fusion of polynomial trend predictions $f_{\text{trend}}(t)$ and physics-informed predictions $f_{\text{physics}}(t)$:

\begin{equation}
f_{\text{hybrid}}(t) = W\bigl[ f_{\text{trend}}(t), f_{\text{physics}}(t) \bigr],
\end{equation}

where $W$ denotes a learned combination operator. Without denoising, the physics-informed contribution $f_{\text{physics}}(t)$ carries significant high-frequency noise, diminishing its relative weight in minimizing forecast error. With the addition of the wavelet denoising layer $D_\psi$, we instead obtain:

\begin{equation}
f_{\text{denoised}}(t) = D_\psi\bigl(f_{\text{hybrid}}(t)\bigr),
\end{equation}

where $D_\psi$ applies wavelet-domain soft thresholding. By filtering out frequencies inconsistent with physically plausible reservoir behavior, the physics-based signal's alignment with observed targets improves, increasing its utility in the final forecast fusion:

\begin{equation}
\text{Var}\bigl(f_{\text{physics}}(t) - y(t)\bigr)_{\text{denoised}} < \text{Var}\bigl(f_{\text{physics}}(t) - y(t)\bigr)_{\text{raw}},
\end{equation}

where $y(t)$ denotes actual measured production data. This variance reduction facilitates a greater emphasis on the physics-informed predictions, shifting fusion weights toward physically motivated explanations.

Thus, the observed shift from a 30\% physics-informed contribution before denoising to 40\% afterward is consistent with the theoretical expectation of improved physical alignment, interpretability, and noise reduction inherent in wavelet-based filtering.


'''