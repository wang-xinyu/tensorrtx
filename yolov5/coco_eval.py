from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#initialize COCO ground truth api
cocoGt = COCO("instances_val2017.json")

#initialize COCO detections api
cocoDt = cocoGt.loadRes("build/yolov5_coco_eval.json")

imgIds=sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()