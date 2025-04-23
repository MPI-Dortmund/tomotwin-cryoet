import argparse
import numpy as np
import json

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_positions', type=str, required=True, help='Path to file containing ground truth positions')
    parser.add_argument('--candidate_positions', type=str, required=True, help='Path to file containing candidate positions')
    parser.add_argument('--beta', type=int, default=6, required=True, help='Value of beta for fbeta score')
    parser.add_argument('--boxsize', type=int, default=37, help='Size of the box')
    return parser

def calculate_precision_recall(gt_positions, candidate_positions, boxsize, beta):
    iou_values = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for gt_point, candidate_point in zip(gt_positions, candidate_positions):
        gt_box = np.array([gt_point - boxsize/2, gt_point - boxsize/2, gt_point + boxsize/2, gt_point + boxsize/2])
        candidate_box = np.array([candidate_point - boxsize/2, candidate_point - boxsize/2, candidate_point + boxsize/2, candidate_point + boxsize/2])
        intersection = np.maximum(np.minimum(gt_box[2], candidate_box[2]) - np.maximum(gt_box[0], candidate_box[0]), 0) * np.maximum(np.minimum(gt_box[3], candidate_box[3]) - np.maximum(gt_box[1], candidate_box[1]), 0)
        union = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) + (candidate_box[2] - candidate_box[0]) * (candidate_box[3] - candidate_box[1]) - intersection
        iou = intersection / union if union != 0 else 0
        iou_values.append(iou)
        if iou > 0.6:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
    fbeta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if precision + recall != 0 else 0

    return precision, recall, true_positives, false_positives, false_negatives, fbeta_score

def main():
    parser = create_argparser()
    args = parser.parse_args()

    gt_positions_df = pd.read_csv('ground_truth_positions.txt', sep=',', header=None)
    gt_positions_df.columns = ["class", "X", "Y", "Z", "R1", "R2", "R3"]
    gt_positions = np.array(gt_positions_df[["X", "Y", "Z"]].values)

    candidate_positions = np.loadtxt(args.candidate_positions)

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