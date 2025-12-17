import argparse, os, json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np


def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    areaA = (xa2 - xa1)*(ya2 - ya1)
    areaB = (xb2 - xb1)*(yb2 - yb1)

    union = areaA + areaB - inter
    if union == 0: return 0
    return inter / union


def load_gt(label_file, img_w, img_h):
    result=[]
    if not Path(label_file).exists(): return result
    with open(label_file) as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w/2)*img_w
            y1 = (yc - h/2)*img_h
            x2 = (xc + w/2)*img_w
            y2 = (yc + h/2)*img_h
            result.append([x1,y1,x2,y2])
    return result


def evaluate_split(model, img_dir, lbl_dir, iou_thr=0.5):
    imgs = list(Path(img_dir).glob("*"))
    TP=FP=FN=TN=0
    for img_path in imgs:
        img = Image.open(img_path)
        w,h = img.size

        gts = load_gt(lbl_dir/f"{img_path.stem}.txt", w,h)

        preds_raw = model.predict(str(img_path), conf=0.001)[0]
        preds=[]
        for b in preds_raw.boxes.xyxy.cpu().numpy():
            preds.append([b[0],b[1],b[2],b[3]])

        matched=set()
        for p in preds:
            best_i=0
            best_gt=-1
            for i,g in enumerate(gts):
                if i in matched: continue
                ii = iou(p,g)
                if ii>best_i:
                    best_i=ii; best_gt=i
            if best_i>=iou_thr:
                TP+=1
                matched.add(best_gt)
            else:
                FP+=1

        FN += (len(gts)-len(matched))
        if not preds and not gts:
            TN+=1

    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec = TP/(TP+FN) if TP+FN>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0

    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"precision":prec,"recall":rec,"f1":f1,"accuracy":acc}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    args=parser.parse_args()

    run=Path(args.run_dir)
    weights = sorted((run/"weights").glob("epoch*.pt"), key=lambda x: int(x.stem.replace("epoch","")))

    results=[]

    for w in weights:
        print("Evaluating:", w.name)
        model = YOLO(str(w))

        train_imgs = Path(args.data_dir)/"train"/"images"
        train_lbls = Path(args.data_dir)/"train"/"labels"
        val_imgs = Path(args.data_dir)/"val"/"images"
        val_lbls = Path(args.data_dir)/"val"/"labels"

        train_m = evaluate_split(model, train_imgs, train_lbls)
        val_m = evaluate_split(model, val_imgs, val_lbls)

        results.append({"epoch":int(w.stem.replace("epoch","")),
                        "train":train_m,
                        "val":val_m})

        with open(run/"per_epoch_metrics.json","w") as f:
            json.dump(results,f,indent=2)

    print("Saved:", run/"per_epoch_metrics.json")
