from torchvision.models import VisionTransformer


def vit_base_patch16():
    hidden_dim = 768
    model = VisionTransformer(224, 16, 12, 12, hidden_dim, 4 * hidden_dim)
    return model


def vit_large_patch16():
    hidden_dim = 1024
    model = VisionTransformer(224, 16, 24, 16, hidden_dim, 4 * hidden_dim)
    return model


def vit_huge_patch14():
    hidden_dim = 1280
    model = VisionTransformer(224, 14, 32, 16, hidden_dim, 4 * hidden_dim)
    return model
