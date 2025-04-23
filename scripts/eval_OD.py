import argparse
import numpy as np
import json
import pandas as pd

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_positions', '-gt', type=str, required=True, help='Path to file containing ground truth positions')
    parser.add_argument('--candidate_positions', '-c', type=str, required=True, help='Path to file containing candidate positions')
    parser.add_argument('--beta', type=int, default=6, required=True, help='Value of beta for fbeta score')
    parser.add_argument('--boxsize', type=int, default=37, help='Size of the box')
    return parser

def calculate_precision_recall(gt_positions, candidate_positions, boxsize, beta):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_candidates = set()

    for gt_point in gt_positions:
        gt_box = np.array([
            [gt_point[0] - boxsize/2, gt_point[1] - boxsize/2, gt_point[2] - boxsize/2],
            [gt_point[0] + boxsize/2, gt_point[1] + boxsize/2, gt_point[2] + boxsize/2]
        ])
        found_match = False

        for i, candidate_point in enumerate(candidate_positions):
            if i in matched_candidates:
                continue

            candidate_box = np.array([
                [candidate_point[0] - boxsize/2, candidate_point[1] - boxsize/2, candidate_point[2] - boxsize/2],
                [candidate_point[0] + boxsize/2, candidate_point[1] + boxsize/2, candidate_point[2] + boxsize/2]
            ])

            x1 = max(gt_box[0, 0], candidate_box[0, 0])
            y1 = max(gt_box[0, 1], candidate_box[0, 1])
            z1 = max(gt_box[0, 2], candidate_box[0, 2])
            x2 = min(gt_box[1, 0], candidate_box[1, 0])
            y2 = min(gt_box[1, 1], candidate_box[1, 1])
            z2 = min(gt_box[1, 2], candidate_box[1, 2])

            intersection_volume = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)

            gt_volume = (gt_box[1, 0] - gt_box[0, 0]) * (gt_box[1, 1] - gt_box[0, 1]) * (gt_box[1, 2] - gt_box[0, 2])
            candidate_volume = (candidate_box[1, 0] - candidate_box[0, 0]) * (candidate_box[1, 1] - candidate_box[0, 1]) * (candidate_box[1, 2] - candidate_box[0, 2])
            union_volume = gt_volume + candidate_volume - intersection_volume

            iou = intersection_volume / union_volume if union_volume != 0 else 0

            print(f"GT Point: {gt_point}, Candidate Point: {candidate_point}, IoU: {iou}")  # Debug print

            if iou > 0.6:
                true_positives += 1
                matched_candidates.add(i)
                found_match = True
                break

        if not found_match:
            false_negatives += 1


    # print(f"Length of candidate_positions: {len(candidate_positions)}")
    # print(f"Length of matched_candidates: {len(matched_candidates)}")

    false_positives = len(candidate_positions) - len(matched_candidates)

    # print(f"Matched Candidates: {matched_candidates}")  # Debug print
    # print(f"False Positives: {false_positives}")  # Debug print

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
    fbeta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if precision + recall != 0 else 0

    return precision, recall, true_positives, false_positives, false_negatives, fbeta_score

def main():
    parser = create_argparser()
    args = parser.parse_args()

    gt_positions_df = pd.read_csv(args.gt_positions, sep=',', header=None)
    gt_positions_df.columns = ["class", "X", "Y", "Z", "R1", "R2", "R3"]
    gt_positions = np.array(gt_positions_df[["X", "Y", "Z"]].values)

    candidate_positions_df = pd.read_csv(args.candidate_positions, sep=',', header=None)
    candidate_positions = np.array(candidate_positions_df.values).astype(float)

    precision, recall, true_positives, false_positives, false_negatives, fbeta_score = calculate_precision_recall(gt_positions, candidate_positions, args.boxsize, args.beta)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'True Positives: {true_positives}')
    print(f'False Positives: {false_positives}')
    print(f'False Negatives: {false_negatives}')
    print(f'FBeta Score: {fbeta_score:.4f}')

    with open('evaluation.json', 'w') as f:
        json.dump({
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'fbeta_score': fbeta_score
        }, f)

if __name__ == '__main__':
    main()