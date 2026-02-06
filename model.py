import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# DAFTAR PENYAKIT
DISEASES = [
    "Early Blight leaf disease",
    "Late Blight leaf disease",
    "Leaf Mold disease",
    "Bacterial Spot leaf disease",
    "Leaf Curl disease",
    "Healthy leaf"
]

# DATABASE OBAT & SOLUSI
DISEASE_TREATMENTS = {
    "Early Blight leaf disease": {
        "chemical": [
            {"name": "Chlorothalonil", "link": "https://www.tokopedia.com/search?st=product&q=Chlorothalonil"},
            {"name": "Mancozeb", "link": "https://www.tokopedia.com/search?st=product&q=Mancozeb"},
            {"name": "Copper-based fungicides", "link": "https://www.tokopedia.com/search?st=product&q=Copper+fungicide"}
        ],
        "organic": [
            {"name": "Neem oil spray", "link": "https://www.tokopedia.com/search?st=product&q=Neem+oil"},
            {"name": "Baking soda solution", "link": "https://www.tokopedia.com/search?st=product&q=Baking+soda"},
            {"name": "Compost tea", "link": "https://www.tokopedia.com/search?st=product&q=Compost+tea"}
        ],
        "prevention": [
            "Rotate crops yearly",
            "Avoid overhead watering",
            "Remove infected leaves promptly"
        ]
    },
    "Late Blight leaf disease": {
        "chemical": [
            {"name": "Metalaxyl", "link": "https://www.tokopedia.com/search?st=product&q=Metalaxyl"},
            {"name": "Mancozeb", "link": "https://www.tokopedia.com/search?st=product&q=Mancozeb"},
            {"name": "Copper oxychloride", "link": "https://www.tokopedia.com/search?st=product&q=Copper+oxychloride"}
        ],
        "organic": [
            {"name": "Potassium bicarbonate spray", "link": "https://www.tokopedia.com/search?st=product&q=Potassium+bicarbonate"},
            {"name": "Neem oil", "link": "https://www.tokopedia.com/search?st=product&q=Neem+oil"},
            {"name": "Baking soda solution", "link": "https://www.tokopedia.com/search?st=product&q=Baking+soda"}
        ],
        "prevention": [
            "Use resistant varieties",
            "Improve air circulation",
            "Avoid wetting foliage"
        ]
    },
    "Leaf Mold disease": {
        "chemical": [
            {"name": "Copper fungicides", "link": "https://www.tokopedia.com/search?st=product&q=Copper+fungicide"},
            {"name": "Chlorothalonil", "link": "https://www.tokopedia.com/search?st=product&q=Chlorothalonil"}
        ],
        "organic": [
            {"name": "Neem oil", "link": "https://www.tokopedia.com/search?st=product&q=Neem+oil"},
            {"name": "Potassium bicarbonate spray", "link": "https://www.tokopedia.com/search?st=product&q=Potassium+bicarbonate"}
        ],
        "prevention": [
            "Increase spacing between plants",
            "Provide good ventilation",
            "Remove affected leaves"
        ]
    },
    "Bacterial Spot leaf disease": {
        "chemical": [
            {"name": "Copper hydroxide", "link": "https://www.tokopedia.com/search?st=product&q=Copper+hydroxide"},
            {"name": "Streptomycin sulfate (if allowed)", "link": "https://www.tokopedia.com/search?st=product&q=Streptomycin"}
        ],
        "organic": [
            {"name": "Neem oil", "link": "https://www.tokopedia.com/search?st=product&q=Neem+oil"},
            {"name": "Baking soda solution", "link": "https://www.tokopedia.com/search?st=product&q=Baking+soda"},
            {"name": "Hydrogen peroxide solution", "link": "https://www.tokopedia.com/search?st=product&q=Hydrogen+peroxide"}
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Avoid working with wet plants",
            "Remove infected plant debris"
        ]
    },
    "Leaf Curl disease": {
        "chemical": [
            {"name": "Mancozeb-based fungicides", "link": "https://www.tokopedia.com/search?st=product&q=Mancozeb"},
            {"name": "Systemic fungicides recommended locally", "link": "https://www.tokopedia.com/search?st=product&q=Systemic+fungicide"}
        ],
        "organic": [
            {"name": "Neem oil spray", "link": "https://www.tokopedia.com/search?st=product&q=Neem+oil"},
            {"name": "Horticultural oil", "link": "https://www.tokopedia.com/search?st=product&q=Horticultural+oil"}
        ],
        "prevention": [
            "Prune infected branches",
            "Avoid overhead watering",
            "Plant resistant varieties"
        ]
    },
    "Healthy leaf": {
        "chemical": [],
        "organic": [],
        "prevention": [
            "Maintain regular plant care",
            "Monitor for pests and diseases"
        ]
    }
}

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    prompts = [f"clear photo of plant leaf with {d}" for d in DISEASES]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    results = sorted(
        zip(DISEASES, probs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return results

# Fungsi gabungan prediksi + lookup obat
def predict_and_treat(image_path):
    results = predict_image(image_path)
    top_disease, confidence = results[0]

    treatment = DISEASE_TREATMENTS.get(top_disease, {
        "chemical": [],
        "organic": [],
        "prevention": []
    })

    return {
        "predicted_disease": top_disease,
        "confidence": confidence,
        "chemical_solution": treatment["chemical"],
        "organic_solution": treatment["organic"],
        "prevention_tips": treatment["prevention"]
    }
