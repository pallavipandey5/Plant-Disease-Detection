def evaluate_custom_metrics(dataset_dicts, predictor, iou_threshold=0.4, detection_threshold=0.5):
    total_detected_gt_disease = 0
    total_valid_detections = 0
    total_true_positives = 0
    total_false_positives = 0
    total_gt_disease_instances = 0

    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        pred_instances = outputs["instances"].to("cpu")

        gt_disease_count = 0
        detected_gt_disease = 0
        true_positives = 0
        false_positives = 0

        for gt_instance in d["annotations"]:
            if gt_instance["category_id"] != 0:
                continue  # Skip non-disease annotations

            gt_polygon = gt_instance["segmentation"][0]
            gt_polygon = [gt_polygon[i:i+2] for i in range(0, len(gt_polygon), 2)]
            gt_disease_count += 1

            iou_max = 0
            pred_masks = pred_instances.pred_masks
            for pred_mask in pred_masks:
                pred_mask = pred_mask.numpy()
                contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) >= 4:
                        pred_polygon = contour.squeeze().tolist()
                        iou = compute_iou(gt_polygon, pred_polygon)
                        iou_max = max(iou_max, iou)  # checking each predicted disease with all the
                        # gt disease to know which one is the same disease with gt

            if iou_max > iou_threshold:
                detected_gt_disease += 1  # it is incrementing depending upon how many
                # disease we have in an image

        if gt_disease_count > 0:
            total_gt_disease_instances += gt_disease_count

        if gt_disease_count > 0 and detected_gt_disease / gt_disease_count >= detection_threshold:
            # if total number of disease with overlapping of th=0.4 <= 20%
            # then increment total valid detection ...to compare for the final metric
            total_valid_detections += 1

            if detected_gt_disease > 0:
                true_positives += 1
            else:
                false_positives += 1

        total_detected_gt_disease += detected_gt_disease
        total_true_positives += true_positives
        total_false_positives += false_positives

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_detected_gt_disease / total_gt_disease_instances if total_gt_disease_instances > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


precision, recall, f1_score = evaluate_custom_metrics(dataset_dicts_T, predictor, iou_threshold=0.4, detection_threshold=0.2)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
