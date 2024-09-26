from src.agentic_rag.llms.llama import BaseLLM
from src.agentic_rag.embeddings.huggingface_embeddings import HuggingFaceEmbedding

import os

import logging
import sys
from pprint import pprint

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document
)
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
# from llama_index.text_splitter import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)
import os
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core import Settings

os.environ['OPENAI_API_KEY'] = ''
system_prompt = """You are a Q&A assistant.
    Your goal is to answer questions as accurately as possible based on the instructions and context providedin with detail
    and please provide reference for your answers and do not generate information on yourself and do not generate incomplete sentences. """



# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
sentence_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

embed_model_name = "sentence-transformers/all-mpnet-base-v2"
# embed_model_name = "path/to/folder"
indexer_db = '/Users/asma/rag/willi_db/'

# print(embed_model_name)
# print(indexer_db)

embed_model = HuggingFaceEmbedding(model_name=embed_model_name, max_length=512, embed_dim=768)
Settings.embed_model = embed_model

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
# ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=sentence_node_parser, callback_manager=callback_manager)


# root = "/content/drive/MyDrive/rag/data/"
root = "/Users/asma/rag/willi/"
# print(root)
# documents = []
from tqdm import tqdm
first_time = True
counter = 1
sentence_index = None


for path, subdirs, files in os.walk(root):
    for name in tqdm(files):
        print(name, counter)
        if os.path.join(path, name).endswith(".pdf"):
            
            if counter % 10 == 0:
                print('saving started ....')
                sentence_index.storage_context.persist(persist_dir=indexer_db)
                sentence_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=indexer_db),
                llm=llm, embed_model=embed_model, callback_manager=callback_manager
                )
            if first_time:
                print(counter , ':' , os.path.join(path, name))
                document = SimpleDirectoryReader(input_files=[os.path.join(path, name)]).load_data()
                nodes = sentence_node_parser.get_nodes_from_documents(document)
                sentence_index = VectorStoreIndex(nodes, embed_model=embed_model, callback_manager=callback_manager)
                first_time = False
            else:
                print(counter , ':' ,os.path.join(path, name))
                document = SimpleDirectoryReader(input_files=[os.path.join(path, name)]).load_data()
                node = sentence_node_parser.get_nodes_from_documents(document)
                sentence_index.insert_nodes(node)
                counter += 1

sentence_index.storage_context.persist(persist_dir=indexer_db)