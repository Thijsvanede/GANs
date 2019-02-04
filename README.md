# GANs
Implementation of various Generative Adversarial Networks (GANs) in Keras.

## Installation
The repository can be cloned using
```bash
git clone https://github.com/Thijsvanede/GANs.git
```

## Organisation
This repository is organised as follows:
```bash
GANs
├── data
├── images
├── LICENSE
├── README.md
├── saved
└── src
    └── GAN.py
```

## GAN
The `GAN` class is used for training and predicting of GANs. This class also acts as a superclass for all other GAN implementations. The class is easily extendible by overriding several key methods. This is described in the section [GAN extensions](#gan-extensions)

### API

#### __init__()
```
__init__(self, dim_input_g=100,
                   dim_input_d=(28, 28),
                   optimizer=Adam(beta_1=0.5)):

    Generative Adversarial Network.

        Parameters
        ----------
        dim_input_g : int, default=100
            Dimension of generator input.

        dim_input_d : tuple, default=(28, 28)
            Dimension of discriminator input.

        optimizer : keras.optimizer, default=Adam(beta_1=0.5)
            Optimiser to use for training.
```

#### train()
```
train(self, X_train, iterations=1000,
                         batch_size=64,
                         k=1,
                         sample_interval=100,
                         verbose=True):

    Train the Generative Adversarial Network with given samples.

        Parameters
        ----------
        X_train : np.array of shape=(n_samples, dim_input_d)
            Real samples to train with.

        iterations : int, default=1000
            Number of iterations to use for training.

        batch_size : int, default=64
            Number of samples in each batch, 1 batch is used per iteration.

        k : int, default=1
            Number of discriminator updates per generator update.

        sample_interval : int, default=100
            Iteration interval at which to output randomly generated
            results of generator.

        verbose : boolean, default=True
            If verbose is set, print current status.
```

#### generate()
```
generate(self, noise=None, amount=5):

    Generate output from given noise.

        Parameters
        ----------
        noise : np.array of shape=(n_samples, dim_input_generator), optional
            If given, generate output from given noise.

        amount : int, default=5
            If no noise is given, generate the amount of output data given
            by this integer.

        Returns
        -------
        result : np.array of shape=(n_samples, dim_output)
            Generated data.
```

#### save()
```
save(self, out_gen, out_dis, out_com):

    Saves weights of GAN to outfile.

        Parameters
        ----------
        out_gen : string
            Path to output file for generator.

        out_dis : string
            Path to output file for discriminator.

        out_com : string
            Path to output file for combined model.
```

#### load()
```
load(self, in_gen, in_dis, in_com):

    Loads weights of GAN from infile.

        Parameters
        ----------
        in_gen : string
            Path to input file for generator.

        in_dis : string
            Path to input file for discriminator.

        in_com : string
            Path to input file for combined model.
```

#### sample_images()
```
sample_images(self, outfile, data=None, width=5, height=5):

    Generate width x height images and write them to outfile.

        Parameters
        ----------
        outfile : string
            Path to outfile to write image to.

        width : int, default=5
            Number of generated images in width of output figure.

        height : int, default=5
            Number of generated images in height of output figure.
```

### GAN Extensions
In order to extend the GAN with custom implementations, one can implement the `build_generator()` and `build_discriminator()` methods of the GAN subclass. These classes should return a `keras.model` of the desired generator and discriminator object respectively.
