import torch.nn
from torch import nn


class HierarchicalLoss(nn.Module):
    def __init__(self, hierarchy, weight_factor=0.5):
        super(HierarchicalLoss, self).__init__()
        self.hierarchy = hierarchy
        self.weight_factor = weight_factor

    def forward(self, logits, targets):
        loss = nn.CrossEntropyLoss()(logits, targets)

        # Add hierarchical penalty
        for parent, children in self.hierarchy.items():
            if targets.item() in [torch.tensor(child) for child in children]:
                loss += self.weight_factor * nn.CrossEntropyLoss()(logits, torch.tensor(children).to(targets.device))

        return loss
    


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-7
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1. - dice
    

'''class HierarchicalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(HierarchicalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true, hierarchy):
        """
        Computes the hierarchical loss.
        
        Args:
        - y_pred (Tensor): Predicted probabilities (batch_size, num_classes).
        - y_true (Tensor): True binary labels (batch_size, num_classes).
        - hierarchy (Tensor): Hierarchical structure matrix (num_classes, num_classes).
        
        Returns:
        - loss (Tensor): Hierarchical loss.
        """
        y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to convert logits to probabilities
        
        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        
        # Compute hierarchical loss
        h_loss = torch.zeros_like(bce_loss)
        for i in range(y_pred.size(0)):
            for j in range(y_pred.size(1)):
                if y_true[i][j] == 1:  # Only consider losses for positive classes
                    ancestors = torch.nonzero(hierarchy[:, j]).squeeze(1)
                    if len(ancestors) > 0:
                        ancestor_loss = torch.mean(bce_loss[i, ancestors])
                        descendant_loss = torch.mean(bce_loss[i, hierarchy[j]])
                        h_loss[i, j] = self.alpha * ancestor_loss + self.beta * descendant_loss
        
        # Average over samples and classes
        loss = torch.mean(h_loss)
        return loss'''