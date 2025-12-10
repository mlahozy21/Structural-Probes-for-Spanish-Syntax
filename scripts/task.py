"""Contains classes describing linguistic tasks of interest on annotated data."""

import numpy as np
import torch

class Task:
  """Abstract class representing a linguistic task mapping texts to labels."""

  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class ParseDistanceTask(Task):
  """Maps observations to dependency parse distances between words."""

  @staticmethod
  def labels(observation):
    """Computes the distances between all pairs of words; returns them as a torch tensor.
    
    VERSION OPTIMIZADA: Usa Floyd-Warshall en lugar de bucles infinitos.
    """
    # 1. Procesar los índices (parsear el formato de Observation)
    head_indices = []
    number_of_underscores = 0
    if hasattr(observation, 'head_indices'):
        raw_heads = observation.head_indices
    else:
        # Fallback si el formato es distinto (tuplas)
        raw_heads = observation[6] # Asumiendo columna 6 si es raw tuple

    for elt in raw_heads:
      if elt == '_':
        head_indices.append(0)
        number_of_underscores += 1
      else:
        # Convertir a int. Ajustar por underscores si los hubiera
        try:
           val = int(elt)
        except ValueError:
           val = 0 
        head_indices.append(val + number_of_underscores)

    seq_len = len(head_indices)
    
    # 2. Inicializar matriz de distancias con Infinito
    # Usamos numpy para velocidad
    dist_matrix = np.full((seq_len, seq_len), np.inf)
    np.fill_diagonal(dist_matrix, 0) # Distancia a sí mismo es 0

    # 3. Llenar conexiones directas (Adyacencia)
    for i, head in enumerate(head_indices):
      if head == 0: continue # 0 es ROOT, no apunta a nadie
      
      head_idx = head - 1 # Convertir de 1-based (CoNLL) a 0-based (Python)
      
      # Protección contra índices fuera de rango o ciclos simples
      if head_idx < 0 or head_idx >= seq_len: continue
      
      # La distancia entre padre e hijo es 1 (bidireccional para distancia sintáctica)
      dist_matrix[i, head_idx] = 1
      dist_matrix[head_idx, i] = 1

    # 4. Algoritmo Floyd-Warshall Vectorizado
    # Esto calcula el camino más corto entre TODOS los pares simultáneamente.
    # Es mucho más rápido que hacerlo par a par.
    for k in range(seq_len):
        dist_matrix = np.minimum(dist_matrix, dist_matrix[:, [k]] + dist_matrix[[k], :])

    # 5. Convertir infinitos a un número alto (para evitar NaNs si hay grafos desconectados)
    dist_matrix[dist_matrix == np.inf] = seq_len + 10

    return torch.tensor(dist_matrix, dtype=torch.float)

class ParseDepthTask:
  """Maps observations to a depth in the parse tree for each word"""

  @staticmethod
  def labels(observation):
    """Computes the depth of each word; returns them as a torch tensor."""
    
    # Procesar índices igual que arriba
    head_indices = []
    number_of_underscores = 0
    if hasattr(observation, 'head_indices'):
        raw_heads = observation.head_indices
    else:
        raw_heads = observation[6]

    for elt in raw_heads:
      if elt == '_':
        head_indices.append(0)
        number_of_underscores += 1
      else:
        try:
           val = int(elt)
        except ValueError:
           val = 0
        head_indices.append(val + number_of_underscores)

    sentence_length = len(head_indices)
    depths = torch.zeros(sentence_length)

    # Calculamos profundidad con un límite de seguridad para evitar bucles infinitos
    for i in range(sentence_length):
      depth = 0
      current = i + 1 # 1-based index
      safety_counter = 0
      
      while True:
        # Si llegamos a ROOT (0) terminamos
        if current == 0:
            break
            
        # Obtenemos el padre
        try:
            current = head_indices[current - 1]
        except IndexError:
            break # Error en datos, salimos
            
        if current != 0:
            depth += 1
            
        # PROTECCIÓN CONTRA CICLOS INFINITOS
        safety_counter += 1
        if safety_counter > sentence_length + 5:
            # Si hemos recorrido más pasos que palabras tiene la frase, hay un ciclo.
            # Cortamos aquí para no colgar el programa.
            break
            
      depths[i] = depth
      
    return depths