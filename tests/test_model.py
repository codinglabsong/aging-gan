import torch
from aging_gan import model


def test_generator_output_shape():
    """Generator preserves input image dimensions."""
    G = model.Generator(ngf=8, n_residual_blocks=1)
    x = torch.randn(2, 3, 64, 64)
    y = G(x)
    assert y.shape == x.shape


def test_discriminator_output_shape():
    """Discriminator outputs a single logit per image."""
    D = model.Discriminator(ndf=8)
    x = torch.randn(2, 3, 64, 64)
    out = D(x)
    assert out.shape == (2, 1)


def test_initialize_models_types():
    """Model initializer returns correct component classes."""
    G, F, DX, DY = model.initialize_models(ngf=8, ndf=8, n_blocks=1)
    assert isinstance(G, model.Generator)
    assert isinstance(F, model.Generator)
    assert isinstance(DX, model.Discriminator)
    assert isinstance(DY, model.Discriminator)
