import torch
from torch_geometric.data import Data


class HierData(Data):

    def to(self, device: str = 'cpu'):
        if hasattr(self, 'surface'):
            self.surface = self.surface.to(device)
        elif hasattr(self, 'patch'):
            self.patch = self.patch.to(device)

        if hasattr(self, 'surface_to_patch'):
            self.surface_to_patch = self.surface_to_patch.to(device)
        self.backbone = self.backbone.to(device)
        self.mapping = self.mapping.to(device)
        if hasattr(self, 'y') and self.y is not None:
            self.y = self.y.to(device)
        return self

    def __repr__(self):
        reprs = []
        if hasattr(self, 'surface'):
            reprs.append(f"Surface={repr(self.surface).strip()}")
        elif hasattr(self, 'patch'):
            reprs.append(f"Patch={repr(self.patch).strip()}")
        if hasattr(self, 'surface_to_patch'):
            reprs.append(f"Surface2Patch={repr(self.surface_to_patch).strip()}")
        reprs.extend([f"Backbone={repr(self.backbone).strip()}",
                      f"Mapping={self.mapping.shape}"])
        return ", ".join(reprs)


class ComplexData(Data):

    def __init__(self, protein: Data, protein2: Data, protein3: Data, ligand: Data, **kwargs):
        self.protein = protein
        self.protein2 = protein2
        self.protein3 = protein3
        self.ligand = ligand
        self.y = ligand.y if ligand.y is not None else None

    def to(self, device: str):
        self.protein = self.protein.to(device)
        self.protein2 = self.protein2.to(device)
        self.protein3 = self.protein3.to(device)
        self.ligand = self.ligand.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self

    def __repr__(self):
        reprs = [f"Protein=({repr(self.protein).strip()})",
                 f"Ligand=({repr(self.ligand).strip()})"]
        if self.y is not None:
            reprs.append(f"Target={self.y.shape}")
        return ", ".join(reprs)

