import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


model_id = "Qwen/Qwen3.5-9B"

def main():
    print("Loading model in VRAM...")
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    example_array = [
        [
            "a computer screen shows a repair prompt on the screen.",
            "a computer screen with a repair automatically pop up",
            "partial computer screen showing the need of repairs",
            "part of a computer monitor showing a computer repair message.",
            "the top of a laptop with a blue background and dark blue text."
        ],
        [
        "a person is holding a bottle that has medicine for the night time.",
        "a bottle of medication has a white twist top.",
        "night time medication bottle being held by someone",
        "a person holding a small black bottle of night time",
        "a bottle of what appears to be cough syrup held in hand."
        ],
        [
        "a white paper showing an image of black and brown dog",
        "a library book with pictures of two dogs on the cover on a wooden table.",
        "a book with a black and a tan dog walking down a snowy street.",
        "the book cover shows two dogs in the snow",
        "a book cover title dog years with an image of a black and brown dog walking up the street, on the left side it has a due date sticker from a library."
        ],
        [
        "a white box is to the left of a blue box on a wooden table.",
        "a small rectangular red and white box next to a small rectangular blue box on a wooden surface.",
        "two boxes of  medicine, one white and red and the other blue sitting on a table",
        "two boxes that appear to contain medication or eye drops",
        "two boxes of pharmaceutical products left in a table"
        ],
        [
        "close up of a computer monitor that is powered on.",
        "a monitor has a message displayed on it.",
        "pictured here is a screenshot that shows an error message from an app.",
        "computer screen displaying an error saying the display driver is not supported by zoom text.",
        "a screenshot of someone's monitor that is having issues"
        ]
    ]

    #examples_str = json.dumps(example_array, indent=2)
    #instead of json idented we do jump lines between sets and between captions, to make it more clear for the model to understand the structure of the output we want
    examples_str = ""
    for set in example_array:
        for caption in set:
            examples_str += caption + "\n"
        examples_str += "\n"  

    # 3. Construcción del Prompt
    instruction = (
        "Generate 10 sets of 5 captions that could fit similar to this examples of sets of captions. "
        "To each set of caption generated, add a global caption, which contains all the information "
        "of the 5 individual captions on the same sentence. The format has to be strictly in JSON, "
        "an array of arrays of 6 strings (5 captions + 1 global caption). Generate exactly 10 sets.\n\n"
        "Examples:\n"
    )

    user_prompt = instruction + examples_str

    messages = [
        {"role": "system", "content": "You are a helpful data generation assistant. You only output valid JSON. Do not output any conversational text or markdown formatting, just the raw JSON array."},
        {"role": "user", "content": user_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Parámetros de Generación
    # max_tokens: 2048 es suficiente para 10 sets de 6 frases.
    # temperature: 0.7 da buena variabilidad sin perder la estructura lógica.
    N_sets = 10
    sampling_params = SamplingParams(temperature=0.7, max_tokens=150*N_sets)

    # 5. Ejecución (Generación)
    print("Generando captions...")
    outputs = llm.generate([formatted_prompt], sampling_params)

    # 6. Extracción y Limpieza de Resultados
    generated_text = outputs[0].outputs[0].text.strip()

    # Limpieza por si el modelo devuelve el JSON dentro de un bloque markdown ```json
    if generated_text.startswith("```json"):
        print("Detected markdown formatting, cleaning output...")
        generated_text = generated_text[7:]
    if generated_text.startswith("```"):
        print("Detected markdown formatting, cleaning output...")
        generated_text = generated_text[3:]
    if generated_text.endswith("```"):
        print("Detected markdown formatting at the end, cleaning output...")
        generated_text = generated_text[:-3]

    generated_text = generated_text.strip()

    # 7. Verificación final
    try:
        result_json = json.loads(generated_text)
        print("\n¡Generación exitosa! Aquí tienes el primer set como muestra:")
        print(json.dumps(result_json[0], indent=2))
        print(f"\nTotal de sets generados: {len(result_json)}")
            
    except json.JSONDecodeError as e:
        print(f"\nError al decodificar el JSON. El modelo generó algo inesperado:\n{generated_text}")
        print(f"\nDetalle del error: {e}")



if __name__ == "__main__":
    main()