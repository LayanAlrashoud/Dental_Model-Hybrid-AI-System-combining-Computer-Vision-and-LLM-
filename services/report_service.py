from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
from services.prompt_builder import build_dental_prompt

client = OpenAI(api_key=OPENAI_API_KEY)


def compute_findings_summary(detections):
    summary = {
        "Caries": 0,
        "Filling": 0,
        "Crown": 0,
        "Implant": 0,
        "Periapical lesion": 0,
        "Retained root": 0,
        "Root canal filling": 0,
    }

    for d in detections:
        cls = d.get("class_name")
        if cls in summary:
            summary[cls] += 1

    return summary


def compute_status_and_attention(summary):
    disease_classes = ["Caries", "Periapical lesion", "Retained root"]
    treatment_classes = ["Filling", "Crown", "Implant", "Root canal filling"]

    disease_count = sum(summary[c] for c in disease_classes)
    treatment_count = sum(summary[c] for c in treatment_classes)

    # Status
    if disease_count == 0 and treatment_count == 0:
        status = "No Visible Findings"
    elif disease_count > 0 and treatment_count == 0:
        status = "Abnormal Findings"
    elif disease_count == 0 and treatment_count > 0:
        status = "Previously Treated"
    else:
        status = "Mixed Findings"

    # Attention Level
    if summary["Periapical lesion"] > 0 or summary["Retained root"] > 0:
        attention = "High"
    elif summary["Caries"] > 0:
        attention = "Medium"
    elif treatment_count > 0:
        attention = "Low"
    else:
        attention = "Low"

    return status, attention


def generate_dental_report(result_data):
    detections = result_data.get("detections", [])

    # 1) summary
    findings_summary = compute_findings_summary(detections)

    # 2) status + attention
    status, attention_level = compute_status_and_attention(findings_summary)

    # 3) build prompt
    prompt = build_dental_prompt(
        findings_summary=findings_summary,
        status=status,
        attention_level=attention_level
    )

    # 4) call LLM
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )

    report_text = response.output_text

    # 5) structured output
    return {
        "summary": report_text,
        "status": status,
        "attention_level": attention_level,
        "findings_summary": findings_summary
    }