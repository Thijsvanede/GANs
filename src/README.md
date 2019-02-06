# GAN
Implementation of regular Generative Adversarial Network

# BiGAN
Implementation of Bidirectional Generative Adversarial Network

# NoBiGAN
Implementation of Novelty detection GAN - only Predict implementation

# NoBiGAN2
Implementation of NoBiGAN where data is generated from noise + class label

# NoBiGAN3
Implementation of NoBiGAN where both data and class label are generated from noise. Only produces 1 or 2 classes in generator and therefore does not learn properly.

# NoBiGAN4
Equivalent to NoBiGAN3 but enforces class labels by including hybrid tuple (Gd(E(x)), y, E(x)). DOES NOT WORK

# NoBiGAN5
Equivalent to NoBiGAN3 but enforces labels by choosing distribution centres

# NoBiGAN6
Equivalent to NoBiGAN 3 but uses a one-hot encoded representation of labels.
