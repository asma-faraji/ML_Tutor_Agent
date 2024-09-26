from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate


class Engine:


    def __init__(self,indexer_db, similarity_top_k=100, rerank_top_n=5):

        self.sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text")


        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])

        self.sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=indexer_db),
        llm=Settings.llm, embed_model=Settings.embed_model, callback_manager=callback_manager
        )


        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )

        self.sentence_window_engine = self.sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank],llm=Settings.llm
        )

        self.retriever_engine = self.sentence_index.as_retriever(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank],llm=Settings.llm)

        self.chat_engine = self.sentence_index.as_chat_engine(llm=Settings.llm,similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank],chat_mode="simple", verbose = True)

    def query(self, query):
        llm_response =  self.sentence_window_engine.query(query)
        return llm_response



engines = Engine(indexer_db)
print(engines.sentence_index)
print(engines.sentence_window_engine)
print(engines.retriever_engine)
# print(engines.sentence_window_engine.query('Summarize grounding requirements for Light Duty site as per Motorola R56.'))

chat_engine = engines.chat_engine

from llama_index.core.chat_engine import CondensePlusContextChatEngine, CondenseQuestionChatEngine
from llama_index.core.memory import BaseMemory

# cpcc = CondensePlusContextChatEngine.from_defaults(
#         retriever=engines.retriever_engine,
#         llm=Setting.llm,
#         verbose=True,
#     )

cpcc = CondenseQuestionChatEngine.from_defaults(
        query_engine=engines.sentence_window_engine,
        llm=Settings.llm,
        verbose=True,
    )