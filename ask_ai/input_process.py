from llm_access.call_llm_test import call_llm

charts = {
    "Bar Chart": "Used for comparing different categories or showing changes over time. ",
    "Pie Chart": "Used for showing the proportion of each part in relation to the whole. ",
    "Line Graph": "Used for illustrating trends over time or other continuous variables. ",
    "Scatter Plot": "Used for showing the relationship or distribution between two variables. "
}

chart_types = ", ".join(charts.keys())
charts_string = str(charts)


def _normalize_chart_type(raw_output: str) -> str:
    """Normalize model output into one of the supported chart type names.

    Args:
        raw_output (str): Raw text output returned by the model.

    Returns:
        str: A supported chart type, defaulting to ``Bar Chart`` when parsing fails.
    """
    if not raw_output:
        return "Bar Chart"

    normalized = raw_output.strip()
    for chart_name in charts:
        if chart_name.lower() in normalized.lower():
            return chart_name
    return "Bar Chart"


def get_chart_type(question, llm):
    """Select a chart type string based on a natural-language question.

    Args:
        question (str): User question describing the analytical goal.
        llm (Any): Language model instance used to classify chart type.

    Returns:
        str: A prompt fragment starting with `", please draw "` and the selected chart type.
    """
    prompt = (
        question
        + " Based on the task described, which chart type is most suitable: "
        + chart_types
        + ". You can only choose one type without any explanation. "
          "If you are unsure, return Bar Chart. "
          "Output only the chart type name. "
        + charts_string
    )
    graph_type = _normalize_chart_type(call_llm(prompt, llm))
    return ", please draw " + graph_type


if __name__ == "__main__":
    print("This module provides chart-type prompt helpers.")
