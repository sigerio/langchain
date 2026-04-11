import pathlib
import unittest


RAG_PATH = pathlib.Path("tutorials/langchain/rag/rag_demo.py")


class RagSourceStructureTest(unittest.TestCase):
    def test_rag_source_file_exists_in_archived_location(self):
        self.assertTrue(RAG_PATH.exists())

    def test_rag_uses_huggingface_embeddings_not_bm25(self):
        source = RAG_PATH.read_text()
        self.assertIn("InferenceClient", source)
        self.assertIn("BAAI/bge-m3", source)
        self.assertIn("Chroma", source)
        self.assertNotIn("BM25Retriever", source)

    def test_rag_prompt_uses_system_and_human_messages(self):
        source = RAG_PATH.read_text()
        self.assertIn('("human", "{question}")', source)

    def test_chat_model_disables_responses_api_for_compatible_provider(self):
        source = RAG_PATH.read_text()
        self.assertIn("use_responses_api=False", source)


if __name__ == "__main__":
    unittest.main()
