import cv2

import torch
from models.transformer import Transformer
from models.position_encoding import PositionEmbeddingSine
from models.backbone import Backbone, Joiner
from models.detr import DETR
import torchvision.transforms as T
from PIL import Image
import struct

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def build_backbone():
    N_steps = 256 // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    train_backbone = True
    return_interm_layers = False
    backbone = Backbone('resnet50', train_backbone, return_interm_layers, False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def gen_wts(model, filename):
    f = open(filename + '.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys()) + 72))
    for k, v in model.state_dict().items():
        if 'in_proj' in k:
            dim = int(v.size(0) / 3)
            q_weight = v[:dim].reshape(-1).cpu().numpy()
            k_weight = v[dim:2*dim].reshape(-1).cpu().numpy()
            v_weight = v[2*dim:].reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k + '_q', len(q_weight)))
            for vv in q_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

            f.write('{} {} '.format(k + '_k', len(k_weight)))
            for vv in k_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

            f.write('{} {} '.format(k + '_v', len(v_weight)))
            for vv in v_weight:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')
        else:
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
            f.write('\n')
    f.close()

def main():
    num_classes = 91
    device = torch.device('cuda')

    backbone = build_backbone()

    transformer = Transformer(
        d_model=256,
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
    )

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=100,
        aux_loss=True,
    )
    checkpoint = torch.load('./detr-r50-e632da11.pth')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    gen_wts(model, "detr")

    # test
    # with torch.no_grad():
    #     transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     im = Image.open('./image/demo.jpg')
    #     img = transform(im).unsqueeze(0)

    #     img = img.to(device)
    #     res = model(img)

    #     logits = res['pred_logits']
    #     pred_boxes = res['pred_boxes']
    #     out_prob = logits.softmax(-1)[0, :, :-1]
    #     keep = out_prob.max(-1).values > 0.5
    #     label = out_prob[keep].argmax(dim=1)
    #     out_bbox = pred_boxes[0, keep]
    #     out_bbox = out_bbox.to(torch.device('cpu'))
    #     out_bbox = box_cxcywh_to_xyxy(out_bbox)
    #     out_bbox = out_bbox * torch.tensor([640, 480, 640, 480])
    #     image = cv2.imread('./image/demo.jpg')
    #     for ob in out_bbox:
    #         x0 = int(ob[0].item())
    #         y0 = int(ob[1].item())
    #         x1 = int(ob[2].item())
    #         y1 = int(ob[3].item())
    #         cv2.rectangle(image, (x0, y0), (x1, y1), (0,0,255), 1)
        
    #     cv2.imwrite('res.jpg', image)

if __name__ == '__main__':
    main()