def prescreen_references(references, prescreening_pipeline, threshold=0.5):
    """
    Prescreen references using a HuggingFace classifier that returns:
        [{'label': True/False, 'score': float}]
    Keeps only references where label == True and score >= threshold.
    """

    screened = []

    for ref in references:
        text = ref.get("text", "").strip()

        # Skip empty / tiny refs
        if len(text) < 10:
            continue

        try:
            result = prescreening_pipeline(text)
        except Exception as e:
            print("Prescreening error:", e)
            continue

        # Expected format: [{'label': bool, 'score': float}]
        if (
            isinstance(result, list) and len(result) == 1
            and isinstance(result[0], dict)
            and "label" in result[0]
        ):
            label = result[0]["label"]
            score = result[0].get("score", 1.0)

            if label is True and score >= threshold:
                screened.append(ref)

        else:
            print("⚠️ Warning: unexpected model output:", result)

    return screened