from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import GPTSimpleVectorIndex, PromptHelper
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index import LLMPredictor, ServiceContext, LangchainEmbedding, Document
from typing import Optional, List, Mapping, Any
from transformers import (
    pipeline,
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification
)

sa_tokenizer = AutoTokenizer.from_pretrained('distilbert')
sa_model = AutoModelForSequenceClassification.from_pretrained('distilbert')
sa_pipeline = pipeline(task='sentiment-analysis', model=sa_model, tokenizer=sa_tokenizer)

class CustomLLM(LLM):
    model_name = 'tinyroberta'
    qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline(task='question-answering', model=qa_model, tokenizer=qa_tokenizer)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.qa_pipeline({'question': prompt, 'context': prompt})
        return response['answer']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'name_of_model': self.model_name}

    @property
    def _llm_type(self) -> str:
        return 'custom'

llm_predictor = LLMPredictor(llm=CustomLLM())
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
prompt_helper = PromptHelper(max_input_size=384, num_output=256, max_chunk_overlap=20)

# define LLM and service context
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, 
    prompt_helper=prompt_helper,
    embed_model=embed_model
)

# Load the data
# with open('data/_coltrane.txt', 'rb') as f:
#     text = f.read().decode('utf-8')
# documents = [Document(sentence) for sentence in text.split('.')]
# index = GPTSimpleVectorIndex.from_documents(
#     documents, 
#     service_context=service_context
# )
# index.save_to_disk('data/index.json')

index = GPTSimpleVectorIndex.load_from_disk(
    save_path='data/index.json',
    service_context=service_context
)

def query_index(query: str) -> str:
    response = index.query(
        query_str=query,
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,
        mode='embedding',
        similarity_top_k=20
    )
    return response.response.strip()