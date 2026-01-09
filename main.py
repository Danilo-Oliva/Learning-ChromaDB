import chromadb

chroma_client =  chromadb.Client()

#para listar usamos
#chroma_client.list_collections()

collection = chroma_client.create_collection(name="prueba")

print(chroma_client.list_collections())

#añadimos una nueva coleccion a la que ya tenemos
collection.add(
  documents=["Primer Documento", "Segundo Documento", "El Málaga CF ganó la copa en el 2002"],
  metadatas=[{"doc":"teatro"}, {"doc":"cine"}, {"doc":"futbol"}],
  ids=["id1", "id2", "id3"]
)

results = collection.query(
  query_texts="Segundo documento",
  n_results=1
)

print(results)

#Embedding
from chromadb.utils import embedding_functions

#con este vemos cual es la funcion por defecto de embedding (all-MiniLM)
embedding_functions.DefaultEmbeddingFunction()

#con esto vamos a conectarnos a hugging face, es un repositorio de modelos, datos, etc.
sentence_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#ahora vamos a hacer un embedding con este modelo
embedding_vector = sentence_embedding(["La plaza del Zócalo es la plaza principal de la ciudad"])

#añadimos la coleccion con el vector del embedding ya hecho
collection.add(
  embeddings=embedding_vector,#este es el vector que almacena el embedding
  documents=["La plaza del Zócalo es la plaza principal de la ciudad"],
  metadatas=[{"doc":"turismo"}],
  ids=["id4"]
)

#si quiero que siempre se guarde con el mismo modelo de embedding lo puedo definir de la siguiente manera
new_collection = chroma_client.create_collection(
  name="Prueba-embedding",
  embedding_function=sentence_embedding #le estoy diciendo que use ese modelo
)

print(chroma_client.list_collections())

#borramos colección
chroma_client.delete_collection(name="prueba")

print(chroma_client.list_collections())