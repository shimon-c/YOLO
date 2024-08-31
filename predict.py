import yolo
import utils
import torch
from torch.utils.data import DataLoader
import argparse
import config
from utils import cells_to_bboxes
from utils import non_max_suppression


def get_loader(test_csv_path):
    from dataset import YOLODataset
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,           #config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    return test_loader

def show_image(model, loader,  thresh=0.5, iou_thresh=0.5, anchors=None, idx=None):
    model.eval()

    if idx is not None:
        x,y = loader.dataset[idx]
        x = x.unsqueeze(dim=0)
    else:
        x, y = next(iter(loader))
    x = x.to("cuda")
    batch_size = 1
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box


    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        utils.plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

def show_preds(checkpoint_path:str=None, test_csv:str=None):
    model = yolo.YOLO(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    adam = torch.optim.Adam(model.parameters())
    utils.load_checkpoint(checkpoint_file=checkpoint_path, model=model, optimizer=adam, lr=config.LEARNING_RATE)
    loader = get_loader(test_csv_path=test_csv)
    idx = 0
    while True:
        show_image(model=model,loader=loader,anchors=scaled_anchors, idx=idx)
        idx += 1

def parse_args():
    ap = argparse.ArgumentParser('Predict')
    ap.add_argument('--checkpoint_path', type=str, default="", help="Full path of checkpoint")
    ap.add_argument('--test_csv_path', type=str, default="", help="Full path to test dir")
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    show_preds(checkpoint_path=args.checkpoint_path, test_csv=args.test_csv_path)