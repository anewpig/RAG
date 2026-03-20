from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def get_local_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.local_embedding_model_name)


def embed_texts_local(texts: list[str]) -> list[list[float]]:
    model = get_local_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
