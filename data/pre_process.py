from mpi4py import MPI

import sys
import json
import torch
import argparse
import os.path as osp
from tqdm import tqdm
from skimage import io

sys.path.append("..")
from baseline.models.tokenization import BertTokenizer

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


parser = argparse.ArgumentParser()

# Data related settings
parser.add_argument(
    "--data_dir",
    default="./panoptic_narrative_grounding",
    help="Path to data directory",
)

args = parser.parse_args()
args_dict = vars(args)
print("Argument list to program")
print("\n".join(["--{0} {1}".format(arg, args_dict[arg]) for arg in args_dict]))
print("\n\n")

splits = ["train2017", "val2017"]
PATH_TO_DATA_DIR = args.data_dir
PATH_TO_FEATURES_DIR = osp.join(PATH_TO_DATA_DIR, "features")
ann_dir = osp.join(PATH_TO_DATA_DIR, "annotations")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


def compute_mask_IoU(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union, intersection / union


cont = 0
for split in tqdm(splits):

    tqdm.write("LOADING {} ANNOTATIONS".format(split.upper()))
    panoptic = load_json(osp.join(ann_dir, "panoptic_{:s}.json".format(split)))
    images = panoptic["images"]
    images = {i["id"]: i for i in images}
    panoptic_anns = panoptic["annotations"]
    panoptic_anns = {int(a["image_id"]): a for a in panoptic_anns}

    panoptic_pred_path = osp.join(
        PATH_TO_FEATURES_DIR, split, "panoptic_seg_predictions"
    )

    tqdm.write("LOADING {} DATA".format(split.upper()))
    panoptic_narratives = load_json(
        osp.join(PATH_TO_DATA_DIR, "annotations", "png_coco_{:s}.json".format(split))
    )

    length = len(panoptic_narratives)
    start = int(rank * int(length / size))
    end = int((start + (length / size) if (rank + 1) != size else length))
    iterable = range(start, end)
    if rank == 0:
        iterable = tqdm(range(start, end))

    all_dict = []
    tqdm.write("FOMATING {} DATA".format(split.upper()))
    for idx in iterable:

        narr = panoptic_narratives[idx]
        words = tokenizer.basic_tokenizer.tokenize(narr["caption"].strip())
        my_words = []
        word_pieces = [tokenizer.wordpiece_tokenizer.tokenize(w) for w in words]

        segments = narr["segments"]
        narr["boxes"] = []
        narr["noun_vector"] = []

        image_id = int(narr["image_id"])
        panoptic_ann = panoptic_anns[image_id]
        segment_infos = {}
        for s in panoptic_ann["segments_info"]:
            idi = s["id"]
            segment_infos[idi] = s

        nom_count = 0

        for seg in segments:

            utter = seg["utterance"].strip()
            if "n't" in utter.lower():
                ind = utter.lower().index("n't")
                all_words1 = tokenizer.basic_tokenizer.tokenize(utter[:ind])
                all_words2 = tokenizer.basic_tokenizer.tokenize(utter[ind + 3 :])
                all_words = all_words1 + ["'", "t"] + all_words2
            else:
                all_words = tokenizer.basic_tokenizer.tokenize(utter)

            my_words.extend(all_words)

            nom_count = nom_count + 1 if len(seg["segment_ids"]) > 0 else nom_count

            for word in all_words:
                word_pi = word

                if not seg["noun"]:
                    narr["boxes"].append([[0] * 4])
                    narr["noun_vector"].append(0)
                elif len(seg["segment_ids"]) == 0:
                    narr["boxes"].append([[0] * 4])
                    narr["noun_vector"].append(0)
                elif len(seg["segment_ids"]) > 0:
                    ids_list = seg["segment_ids"]
                    nose = []
                    for lab in ids_list:
                        box = segment_infos[int(lab)]["bbox"]
                        nose.append(box)
                    narr["boxes"].append(nose)
                    narr["noun_vector"].append(nom_count)
                else:
                    raise ValueError("Error in data")

        if len(words) == len(narr["boxes"]):

            image_info = images[image_id]
            panoptic_segm = io.imread(
                osp.join(
                    ann_dir,
                    "panoptic_segmentation",
                    split,
                    "{:012d}.png".format(image_id),
                )
            )
            panoptic_segm = (
                panoptic_segm[:, :, 0]
                + panoptic_segm[:, :, 1] * 256
                + panoptic_segm[:, :, 2] * 256 ** 2
            )
            panoptic_ann = panoptic_anns[image_id]

            annotations = [item for sublist in narr["boxes"] for item in sublist]
            panoptic_pred = io.imread(
                osp.join(panoptic_pred_path, "{:012d}.png".format(image_id))
            )[:, :, 0]
            panoptic_pred = torch.tensor(panoptic_pred).long()
            proposals = torch.zeros(
                (
                    panoptic_pred.max() + 1,
                    panoptic_pred.shape[0],
                    panoptic_pred.shape[1],
                )
            )
            proposals = proposals.scatter(0, panoptic_pred.unsqueeze(0), 1).long()
            proposals = proposals[1:]

            unique_annotations = torch.tensor(annotations).unique(dim=0).tolist()
            labels = [[-2 for i in sublist] for sublist in narr["boxes"]]
            if len(proposals) > 0:
                iou_matrix = torch.zeros([len(unique_annotations), len(proposals)])
                for i, a in enumerate(unique_annotations):
                    if a != [0] * 4:
                        segment_info = [
                            s for s in panoptic_ann["segments_info"] if s["bbox"] == a
                        ][0]
                        instance = torch.zeros(
                            [image_info["height"], image_info["width"]]
                        )
                        instance[panoptic_segm == segment_info["id"]] = 1
                        for j, p in enumerate(proposals):
                            _, _, iou_matrix[i, j] = compute_mask_IoU(instance, p)
                        iou, indices = torch.max(iou_matrix, dim=1)
                        indices[iou == 0] = -1
                        ann_mask = [
                            [True if ann == a else False for ann in sublist]
                            for sublist in narr["boxes"]
                        ]
                        labels = [
                            [
                                indices[i].item() if m else l
                                for (m, l) in zip(submask, sublabels)
                            ]
                            for (submask, sublabels) in zip(ann_mask, labels)
                        ]
            else:
                labels = [[-1 for i in sublist] for sublist in narr["boxes"]]
                ann_mask = [
                    [True if ann == [0] * 4 else False for ann in sublist]
                    for sublist in narr["boxes"]
                ]
                labels = [
                    [-2 if m else l for (m, l) in zip(submask, sublabels)]
                    for (submask, sublabels) in zip(ann_mask, labels)
                ]
            narr["labels"] = labels

            del narr["segments"]
            all_dict.append(narr)

        else:
            cont += 1

    tqdm.write("{} DATA FORMATED".format(split.upper()), end="\r")

    all_dict = comm.gather(all_dict, root=0)
    if rank == 0:
        panoptic_narrative_grounding = []
        for d in all_dict:
            panoptic_narrative_grounding.extend(d)
        save_json(
            osp.join(
                PATH_TO_DATA_DIR,
                "annotations",
                "png_coco_{}_dataloader.json".format(split),
            ),
            panoptic_narrative_grounding,
        )

tqdm.write("{} Narratives Excluded".format(cont))

