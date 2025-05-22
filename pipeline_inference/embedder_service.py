from openai import OpenAI
import os
from dotenv import load_dotenv
from config import OPENAI_API_KEY

class EmbedderService:
    def __init__(self):
        self.MODEL = "text-embedding-ada-002"
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def get_embedding(self, text): 
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.MODEL
        )
        return response.data[0].embedding