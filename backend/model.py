import torch
import torch.nn as nn
import timm

class DeepfakeModel(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=1):
        super(DeepfakeModel, self).__init__()
        # Load the backbone without the default head
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        
        # Identification of feature size for efficientnet_b3 is 1536
        feature_size = 1536 
        
        # Simple head as indicated by fc.weight/bias in state_dict
        self.fc = nn.Linear(feature_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, frames, channels, h, w) or (batch, channels, h, w)
        if x.dim() == 5:
            # Video clip: (B, T, C, H, W)
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            features = self.backbone(x) # (B*T, 1536, 10, 10) or similar
            # Global Average Pooling
            features = features.mean(dim=[-2, -1]) # (B*T, 1536)
            features = features.view(b, t, -1) # (B, T, 1536)
            # Temporal pooling: mean
            features = features.mean(dim=1) # (B, 1536)
        else:
            # Single image: (B, C, H, W)
            features = self.backbone(x)
            features = features.mean(dim=[-2, -1])
            
        out = self.fc(features)
        return out

def load_model(path, device='cpu'):
    model = DeepfakeModel()
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model']
    
    # Fix for any missing keys or prefix issues
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
