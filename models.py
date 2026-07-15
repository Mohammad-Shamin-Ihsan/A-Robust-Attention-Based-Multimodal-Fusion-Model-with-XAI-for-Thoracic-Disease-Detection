import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        # Initialize a pre-trained Vision Transformer
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # Strip the standard ImageNet classification head and replace it 
        # with a projection layer to match our shared latent space
        self.vit.heads = nn.Linear(self.vit.heads.in_features, embedding_dim)

    def forward(self, cxr_tensor):
        # cxr_tensor shape: (Batch, 3, 320, 320)
        # Dynamically resize to fit the standard ViT patch grid
        cxr_resized = TF.resize(cxr_tensor, [224, 224], antialias=True)
        
        # Output shape: (Batch, embedding_dim)
        return self.vit(cxr_resized)

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=100, embedding_dim=768):
        super().__init__()
        # A standard feed-forward network to map the 100-dim clinical vector 
        # into the exact same dimensional space as the vision encoder
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevents overfitting on the clinical noise
            nn.Linear(256, embedding_dim)
        )

    def forward(self, clinical_vector):
        # clinical_vector shape: (Batch, 100)
        # Output shape: (Batch, embedding_dim)
        
        return self.network(clinical_vector)
    
class BaselineClassifier(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=5):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding):
        # embedding shape: (Batch, 768)
        # Output shape: (Batch, 5) -> Raw logits for the 5 target labels
        return self.classifier(embedding)