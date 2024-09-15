"""Mock embedding model."""
#  curl -X POST -H "Content-Type: application/json" -d {"text":"Hello"} https://92a4-34-126-120-144.ngrok-free.app/get_embeddings
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
import json

# class Item(BaseModel):
#     text : str


class HuggingFaceEmbedding(BaseEmbedding):
    """Mock embedding.

    Used for token prediction.

    Args:
        embed_dim (int): embedding dimension

    """

    # embed_dim : int

    def __init__(self, embed_dim: int, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(embed_dim=embed_dim, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "google colab embeddings"

    ## send request here
    def _get_vector(self, text: str) -> List[float]:
        # request to the google colab
        import requests

        # The API endpoint
        # env var
        server_url = "https://4635-34-126-120-144.ngrok-free.app"
        url = "{server_url}/get_embeddings".format(server_url=server_url)

        # Data to be sent
        data = {
                'text': text
                }

        # A POST request to the API
        response = requests.post(url, json=data)
        embed = response.json()['embed']
        print(len(embed))

        return embed

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_vector()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_vector()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_vector(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_vector(text)



test = HuggingFaceEmbedding(512)
test._get_query_embedding('hello? how are you?')