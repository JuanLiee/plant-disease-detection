import subprocess
import json

MODEL_NAME = "hf.co/CopyleftCultivars/Mistral7B-NaturalFarmerV4-GGUF:Q4_K_M"

def ask_ollama(disease_name: str) -> str:
    """
    Ask Ollama to explain plant disease, its causes, symptoms, and solutions.
    Always returns text (never empty).
    """

    prompt = f"""
You are an agricultural expert.

The detected plant disease is: {disease_name}

Even if the diagnosis is uncertain or confidence is low,
you MUST still explain based on common cases.

Explain clearly in this format:

Disease Overview:
(short explanation)

Possible Causes:
- cause 1
- cause 2

Common Symptoms:
- symptom 1
- symptom 2

Suggested Solutions:
- organic solution
- chemical solution
- prevention tips

Use simple language.
Do NOT say you are unsure.
Do NOT refuse to answer.
"""

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )

        output = result.stdout.strip()

        # FALLBACK SAFETY
        if not output:
            return (
                "AI could not generate a detailed explanation.\n\n"
                "Possible causes include fungal or bacterial infection.\n"
                "Try improving lighting, reducing moisture, and removing infected leaves."
            )

        return output

    except subprocess.TimeoutExpired:
        return (
            "AI explanation timed out.\n\n"
            "This disease is commonly caused by pathogens or poor environmental conditions.\n"
            "Ensure proper airflow, watering, and plant nutrition."
        )

    except Exception as e:
        return (
            "AI explanation failed due to an internal error.\n\n"
            "General advice:\n"
            "- Remove affected leaves\n"
            "- Avoid overwatering\n"
            "- Use appropriate fungicide if needed"
        )
