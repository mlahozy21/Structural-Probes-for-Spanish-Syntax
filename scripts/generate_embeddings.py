import h5py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys

# Usaremos mBERT (multilingual BERT) porque AnCora es español
MODEL_NAME = "bert-base-multilingual-cased"

def get_word_embeddings(text_file, hdf5_file):
    print(f"Cargando modelo: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        fix_mistral_regex=True
        )
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Procesando {text_file} -> {hdf5_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    with h5py.File(hdf5_file, 'w') as f_out:
        for idx, sentence in tqdm(enumerate(lines), total=len(lines)):
            # Tokenizar preservando el mapeo a palabras originales
            inputs = tokenizer(sentence.split(), is_split_into_words=True, return_tensors="pt", return_offsets_mapping=True)
            offset_mapping = inputs["offset_mapping"]
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Obtenemos hidden states de la última capa (shape: 1, seq_len, 768)
            # Ojo: El paper original a veces usa todas las capas. Aquí usaremos la última por simplicidad
            # Si quieres todas, usa outputs.hidden_states (si configuras output_hidden_states=True)
            hidden_states = outputs.last_hidden_state[0].cpu().numpy() # (seq_len, 768)
            
            # Alinear subwords a palabras originales
            word_ids = tokenizer(
                sentence.split(),
                is_split_into_words=True
            ).word_ids()
            
            merged_embeddings = []
            current_word_vectors = []
            current_word_id = None
            
            for i, w_id in enumerate(word_ids):
                if w_id is None: continue # Skip [CLS], [SEP]
                
                if w_id != current_word_id:
                    if current_word_vectors:
                        # Promediar subwords anteriores
                        merged_embeddings.append(np.mean(current_word_vectors, axis=0))
                    current_word_vectors = []
                    current_word_id = w_id
                
                current_word_vectors.append(hidden_states[i])
            
            # Añadir el último grupo
            if current_word_vectors:
                merged_embeddings.append(np.mean(current_word_vectors, axis=0))
            
            # Guardar en HDF5. Clave debe ser el string del índice
            # El shape final debe ser (num_words, 768)
            f_out.create_dataset(str(idx), data=np.array(merged_embeddings))

if __name__ == "__main__":
    get_word_embeddings(sys.argv[1], sys.argv[2])