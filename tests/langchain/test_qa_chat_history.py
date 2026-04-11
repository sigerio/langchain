import pathlib
import unittest


QA_CHAT_HISTORY_PATH = pathlib.Path(
    "tutorials/langchain/agents/qa_chat_history.py"
)


class QaChatHistorySourceStructureTest(unittest.TestCase):
    def test_source_file_exists_in_archived_location(self):
        self.assertTrue(QA_CHAT_HISTORY_PATH.exists())

    def test_source_uses_huggingface_embeddings_not_openai_embeddings(self):
        source = QA_CHAT_HISTORY_PATH.read_text()
        self.assertIn("InferenceClient", source)
        self.assertIn("BAAI/bge-m3", source)
        self.assertIn("Chroma", source)
        self.assertNotIn("OpenAIEmbeddings", source)

    def test_source_keeps_langgraph_memory_flow(self):
        source = QA_CHAT_HISTORY_PATH.read_text()
        self.assertIn("MemorySaver", source)
        self.assertIn("create_react_agent", source)


if __name__ == "__main__":
    unittest.main()
