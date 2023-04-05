import torch
import torchvision

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
    
    @staticmethod
    def get_lengths(img : torch.Tensor):
        h, w = img.size(2), img.size(3)
        longest_side = torch.max(torch.tensor([h, w], dtype=torch.short).detach())
        resize_value = torch.ceil(longest_side / 32) * 32
        return h, w, resize_value.int().item()
    
    @staticmethod
    def preprocess(img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to('cpu')
        img = img.permute(0,3,1,2)
        img = img.float()  # uint8 to fp16/32
        h, w, resize_value = Wrapper.get_lengths(img)
        padding = torch.zeros((1, 3, resize_value, resize_value))
        padding[:, :, :h, :w] = img
        padding /= 255  # 0 - 255 to 0.0 - 1.0
        return padding
    
    @staticmethod
    def xywh2xyxy(x):
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    @staticmethod
    def _non_max_suppression(pred, orig_img, conf_threshold=0.5, iou_threshold=0.4, max_det=300):
        pred.squeeze_()
        boxes, scores, cls = pred[:4, :].T, pred[4:, :].amax(0), pred[4:, :].argmax(0).to(torch.int)
        keep = scores.argsort(0, descending=True)[:max_det]
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        boxes = Wrapper.xywh2xyxy(boxes)
        candidate_idx = torch.arange(0, scores.shape[0])
        candidate_idx = candidate_idx[scores > conf_threshold]

        boxes, scores, cls = boxes[candidate_idx], scores[candidate_idx], cls[candidate_idx]
        final_idx = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)

        boxes = boxes[final_idx]
        scores = scores[final_idx]
        cls = cls[final_idx]

        boxes[:, [0,2]] = boxes[:, [0,2]].clamp(min=0, max=orig_img.size(2)) # width for x 
        boxes[:, [1,3]] = boxes[:, [1,3]].clamp(min=0, max=orig_img.size(1)) # height for y
                
        return torch.cat([boxes, scores.unsqueeze(1), cls.unsqueeze(1)], dim=1)

    @staticmethod
    def postprocess(pred, orig_img):
        result = Wrapper._non_max_suppression(pred, orig_img)
        return result

    def forward(self, imgs):
        orig_img = imgs.clone()
        imgs = Wrapper.preprocess(imgs)
        preds = self.model(imgs)
        result = Wrapper.postprocess(preds[0], orig_img)
        return result