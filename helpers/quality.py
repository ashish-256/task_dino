import torch
from torchvision.transforms import functional as F
from PIL import Image
import timm
import joblib

class QualityChecker:
    def __init__(self, clf_path='./weights/quickqual_dn121_512.pkl', device='cuda'):
        self.device = torch.device(device)
        self.model = timm.create_model('densenet121.tv_in1k', pretrained=True, num_classes=0).to(self.device)
        self.model.eval()
        self.clf = joblib.load(clf_path)

    def check_quality(self, pil_img: Image.Image):
        # Resize to 512 internal for model evaluation requirements
        img = F.to_tensor(F.resize(pil_img, 512))
        img = F.normalize(img, [0.5] * 3, [0.5] * 3).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast(device_type=self.device.type):
            with torch.no_grad():
                preds = self.model(img)
        
        preds = preds.detach().cpu().numpy()
        prob = self.clf.predict_proba(preds)[0]
        
        # Returns (is_bad)
        return (prob[2] > 0.98)