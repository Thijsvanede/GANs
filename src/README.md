# GAN
The file `GAN.py` implementations a regular Generative Adversarial Network. This is the vanilla implementation of the Extensible GAN framework and can be extended to include other implementations.

The GAN has functions for building each different component indicated by the `build_<component_name>` functions. These can be overwritten in order to change the architecture of the GAN. The main functionality offered by the GAN are given by the following functions:
 * `train` trains the GAN networks with the given training data.
 * `generate` generates data samples from a latent input.
 * `save` saves the GAN models to file for reuse.
 * `load` loads the GAN models from file for reuse.
 * `sample_images` generates some random sample images from noise.

All files contain a `main` method which executes the implementation of the GAN variant and prints the evaluation results. Note that the regular GAN implementation only creates and saves the model, it does not print any evaluation.

**For a full API overview we refer to the code documentation.**
```bash
pydoc3 -b
```

All extensions of the vanilla GAN are discussed in the remainder of this document.

## BiGAN
The file `BiGAN.py` implementations a Bidirectional Generative Adversarial Network. The BiGAN extends the regular GAN implementation with an Encoder. Because it also has an encoder it adds functions for visualising the latent representation of data. The added functionality of the BiGAN includes:
 * `build_encoder` builds the encoder network, can be overwritten for custom implementations.
 * `plot_latent` takes in actual data and plots the latent representation.

## Reconstruction-Based AD-GAN
The file `RB_AD_GAN.py` contains the implementation of the Reconstruction-Based Anomaly Detection GAN. This object extends the BiGAN implementation with a method to predict anomalies. It subsequently applies the BiGAN's encoder and generator; computes the MSE loss between original and reconstructed data; and makes a prediction based on a given threshold.

The Reconstruction-Based GAN adds the following methods on top of the BiGAN implementation:
 * `predict` predicts whether samples are normal `1` or anomalous `-1` based on reconstruction performance and a given threshold.

## Class-Based AD-GAN
The file `CB_AD_GAN.py` contains the implementation of the Class-Based Anomaly Detection GAN. This object implements the Class-Based AD-GAN as discussed in the report.

The Class-Based AD-GAN adds the following methods on top of the BiGAN implementation:
 * `build_generator_label` builds the label generator network, can be overwritten for custom implementations.
 * `predict` predicts whether samples are normal `1` or anomalous `-1` based on IsolationForest detection in the latent space.
 * `generate_samples_class` generates samples for each trained class.

## PCA Anomaly Detection
The file `pca_anomaly_detection.py` implementations the PCA based Anomaly Detection as discussed in the report.

It implements the following methods:
 * `fit` fits the detector with normal data.
 * `predict` predicts whether samples are normal `1` or anomalous `-1` based on IsolationForest detection after PCA dimensionality reduction.
 * `fit_predict` subsequently applies `fit` and `predict` on the given dataset.

## Utils
The file `utils.py` contain some auxilliary methods shared among the different GAN implementations.

The three methods in utils are the following:
 * `split` splittings data into known and unknown samples.
 * `scale` rescales the data to a given interval.
 * `evaluate` prints an evaluation report based on the true and predicted labels.
