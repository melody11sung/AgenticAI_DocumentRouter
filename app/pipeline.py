from llama_index.core import Settings, SummaryIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.storage import StorageContext
from llama_index.core.indices import load_index_from_storage
from llama_index.core.base.response.schema import Response
from embeddingSelector import EmbeddingAwareSelector
import logging
import os
import pickle


logger = logging.getLogger(__name__)


class LlamaIndexPipeline:

    def __init__(self):
        self.cache_dir = "cache"
        self.llm = OpenAI(model="gpt-3.5-turbo")
        
        # Use a local cache directory for the embedding model
        embidding_cache_dir = os.path.join(self.cache_dir, "embeddings")
        os.makedirs(embidding_cache_dir, exist_ok=True)
        
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=embidding_cache_dir
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.summary_tool = None
        self.search_tool = None
        self.tool_executer = None


    def load_data(self, docs_dir):
        # Check if cached nodes exist
        cache_file = os.path.join(self.cache_dir, "nodes.pkl")
        if os.path.exists(cache_file):
            logger.info("Loading cached nodes...")
            with open(cache_file, 'rb') as f:
                nodes = pickle.load(f)
            logger.info(f"Loaded {len(nodes)} cached nodes")
            return nodes

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("Processing documents (this may take a while)...")
        reader = SimpleDirectoryReader(input_dir=docs_dir)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents from {docs_dir}")

        splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info(f"Split into {len(nodes)} nodes")

        # Cache the nodes
        logger.info("Caching nodes for future use...")
        with open(cache_file, 'wb') as f:
            pickle.dump(nodes, f)

        return nodes


    def build(self, nodes):
        # Check if cached indices exist
        summary_cache_dir = os.path.join(self.cache_dir, "summary_index")
        vector_cache_dir = os.path.join(self.cache_dir, "vector_index")

        if os.path.exists(summary_cache_dir) and os.path.exists(vector_cache_dir):
            logger.info("Loading cached indices...")
            summary_storage_context = StorageContext.from_defaults(persist_dir=summary_cache_dir)
            vector_storage_context = StorageContext.from_defaults(persist_dir=vector_cache_dir)
            
            summary_index = load_index_from_storage(summary_storage_context)
            vector_index = load_index_from_storage(vector_storage_context)
            logger.info("Loaded cached indices")
        else:
            logger.info("Building indices (this may take a while)...")
            summary_index = SummaryIndex(nodes)
            vector_index = VectorStoreIndex(nodes)
            
            # Cache the indices
            logger.info("Caching indices...")
            summary_index.storage_context.persist(persist_dir=summary_cache_dir)
            vector_index.storage_context.persist(persist_dir=vector_cache_dir)

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            include_text=True,
            use_async=True
        )
        vector_query_engine = vector_index.as_query_engine(similarity_top_k=2)

        self.summary_tool = QueryEngineTool.from_defaults(
            name="summary_tool",
            query_engine=summary_query_engine,
            description=(
                "Summarizes the entire document. Use for high-level understanding."
            )                         
        )

        self.search_tool = QueryEngineTool.from_defaults(
            name="search_tool",
            query_engine=vector_query_engine,
            description=(
                "If the query can be answered by searching the document, use this tool."
                "If the query is out of scope of the document, use the action tool."
            )
        )

        # In case the query cannot be answered by the summary or search tool,
        # the action tool will be used to take extra action with outside API
        self.action_tool = QueryEngineTool.from_defaults(
            name="action_tool",
            query_engine=DummyActionQueryEngine(),
            description=(
                "Handles queries that cannot be answered by the summary or search tool."
            )
        )

        self.tool_executer = RouterQueryEngine(
            selector=EmbeddingAwareSelector(llm=self.llm, vector_index=vector_index, embed_model=self.embed_model),
            query_engine_tools=[self.summary_tool, self.search_tool, self.action_tool],
            verbose=True
            )

    def clear_cache(self):
        """Clear all cached data to force rebuilding"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info("Cache cleared")


class DummyActionQueryEngine():
    def __init__(self):
        pass

    def query(self, query: str) -> Response:
        return Response(response="Dummy action triggered")


