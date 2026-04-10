def build_dental_prompt(findings_summary, status, attention_level):
    """
    Build a structured prompt for dental report generation based on model findings.
    """

    findings_text = ""
    for key, value in findings_summary.items():
        if value > 0:
            findings_text += f"- {key}: {value}\n"

    if findings_text == "":
        findings_text = "No significant findings detected."

    prompt = f"""
You are a professional AI dental assistant.

A dental AI model analyzed a dental image and detected the following findings:

{findings_text}

Overall status: {status}
Attention level: {attention_level}

Your task:
1. Write a clear and professional dental image analysis summary.
2. Interpret the findings (e.g., caries, fillings, crowns, implants).
3. Mention if findings may indicate active issues or previous treatments.
4. Provide short and appropriate recommendations.
5. Clearly state that this is an AI-generated preliminary assessment, not a medical diagnosis.

Keep the response:
- Clear
- Concise
- Professional
Keep the response under 120 words.
Avoid repetition.

Return the response in plain English.
Do NOT use markdown formatting such as ### or **.
Write in normal paragraphs only.
"""

    return prompt