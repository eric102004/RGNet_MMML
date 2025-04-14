"""
Modified from https://github.com/Soldelli/MAD/blob/48f7e1325b7b36b2e7ccc30dfb509880d65b89a2/baselines/VLG-Net/lib/engine/evaluation.py

Script to evaluate performance of any model for MAD
"""

import torch
import tqdm
import terminaltables
import argparse
import json
from utils.basic_utils import load_jsonl


def display_results(results, thresholds, topK, title=None):
    display_data = [
        [f"Rank@{ii}\nmIoU@{jj:.1f}" for ii in topK for jj in thresholds]
    ]
    results *= 100
    display_data.append(
        [
            f"{results[ii][jj]:.02f}"
            for ii in range(len(topK))
            for jj in range(len(thresholds))
        ]
    )
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(thresholds) * len(topK)):
        table.justify_columns[ii] = "center"
    return table.table, display_data

def display_average_iou_or_ndcg(results, topK, score_type, title=None):
    assert score_type in ["aIoU", "nDCG"]
    display_data = [
        [f"{score_type}@{ii}" for ii in topK]
    ]
    #results *= 100
    display_data.append(
        [
            f"{results[ii]:.04f}"
            for ii in range(len(topK))
        ]
    )
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(topK)):
        table.justify_columns[ii] = "center"
    return table.table, display_data


def _iou(candidates, gt):
    start, end = candidates[:, 0].float(), candidates[:, 1].float()
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def _nms(moments, scores, topk, thresh=0.5):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]

    suppressed = torch.zeros_like(moments[:, 0], dtype=torch.bool)
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = _iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
        if i % topk.item() == 0:
            if (~suppressed[:i]).sum() >= topk:
                break

    moments = moments[~suppressed]
    return moments[:topk]


def evaluate_nlq_performance(
        submission, ground_truth, thresholds, topK, match_number=False
):
    pred_qids = set([e["query_id"] for e in submission])
    gt_qids = set([e["query_id"] for e in ground_truth])

    if match_number:
        print(len(pred_qids),len(gt_qids))
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["query_id"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["query_id"] in shared_qids]

    truth_dict = {}
    for d in ground_truth:
        truth_dict[d['query_id']] = d["timestamps"]

    iou_metrics = torch.tensor(thresholds)
    num_iou_metrics = len(iou_metrics)

    recall_metrics = torch.tensor(topK)
    max_recall = recall_metrics.max()
    num_recall_metrics = len(recall_metrics)
    recall_x_iou = torch.zeros((num_recall_metrics, len(iou_metrics)))
    average_x_iou = torch.zeros((num_recall_metrics,))
    ndcg_x_iou = torch.zeros((num_recall_metrics,))
    #results_sample = {}
    results_sample = None

    for k in tqdm.tqdm(submission):
        recall_x_iou_sample = torch.zeros((num_recall_metrics, len(iou_metrics)))
        average_x_iou_sample = torch.zeros((num_recall_metrics,))
        ndcg_x_iou_sample = torch.zeros((num_recall_metrics,))
        
        gt_grounding = torch.tensor(truth_dict[k['query_id']])
        pred_moments = torch.tensor(k["predicted_times"][:max_recall])
        # print("gt_grounding: ",gt_grounding)
        # print("pred_moments: ", pred_moments)
        mious = _iou(pred_moments, gt_grounding)            # (max_recall, )

        mious_len = len(mious)
        bools = mious[:, None].expand(mious_len, num_iou_metrics) > iou_metrics
        # print("mious: ", mious)
        # print("bools: ", bools)
        # exit(1)
        for i, r in enumerate(recall_metrics):
            recall_x_iou_sample[i] += bools[:r].any(dim=0)
            # calculate average_x_iou
            average_x_iou_sample[i] += mious[:r].sum() / r
            # calculate ndcg_x_iou
            ndcg_x_iou_sample[i] += ndcg(mious[:r])

        recall_x_iou += recall_x_iou_sample
        average_x_iou += average_x_iou_sample
        ndcg_x_iou += ndcg_x_iou_sample
        #results_sample[k['query_id']] = {}
        #results_sample[k['query_id']]["recall_x_iou"] = recall_x_iou_sample.tolist()
        #results_sample[k['query_id']]["average_x_iou"] = average_x_iou_sample.tolist()
        #results_sample[k['query_id']]["ndcg_x_iou"] = ndcg_x_iou_sample.tolist()

    recall_x_iou /= len(submission)
    average_x_iou /= len(submission)
    ndcg_x_iou /= len(submission)
    return recall_x_iou, average_x_iou, ndcg_x_iou, results_sample

def ndcg(relevance_scores):
    """
    Calculate the normalized discounted cumulative gain (nDCG) for a sequence of relevance scores.

    Args:
        relevance_scores (torch.Tensor): 1D tensor of relevance scores.

    Returns:
        float: The nDCG value.
    """
    # Sort relevance scores in descending order to get the ideal DCG
    ideal_relevance_scores = torch.sort(relevance_scores, descending=True).values
    # Compute DCG
    dcg = (relevance_scores / torch.log2(torch.arange(1, len(relevance_scores) + 1, dtype=torch.float) + 1)).sum()
    # Compute IDCG
    idcg = (ideal_relevance_scores / torch.log2(torch.arange(1, len(ideal_relevance_scores) + 1, dtype=torch.float) + 1)).sum()
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    # Compute nDCG
    return (dcg / idcg).item()


def main(args):
    print(f"""Reading predictions: {args["model_prediction_json"]}""")
    predictions = load_jsonl(args["model_prediction_json"])

    print(f"""Reading gt: {args["ground_truth_json"]}""")
    ground_truth = load_jsonl(args["ground_truth_json"])
    results = evaluate_nlq_performance(
        predictions, ground_truth, args["thresholds"], args["topK"]
    )
    print(display_results(results, args["thresholds"], args["topK"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground_truth_json",
        required=True,
        help="Ground truth temporal windows",
    )
    parser.add_argument(
        "--model_prediction_json",
        required=True,
        help="Model predicted temporal windows",
    )
    parser.add_argument(
        "--thresholds",
        required=True,
        nargs="+",
        type=float,
        help="Thresholds for IoU computation",
    )
    parser.add_argument(
        "--topK",
        required=True,
        nargs="+",
        type=int,
        help="Top K for computing recall@k",
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
