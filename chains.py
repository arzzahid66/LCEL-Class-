from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever

class SimpleQAChain:
    def __init__(self, model):
        self.model = model

    # Simple QA chain method 
    def QAchain(self, query,template):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            input_dict = {"question": query}
            response = chain.invoke(input_dict)
            return response
        except Exception as e:
            return f"Error executing chain: {str(e)}"


    # Simple conversational chain method
    def Conversational_Chain(self, query, history, template):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            
            # Create input dict with variables matching the template
            input_dict = {
                "HISTORY": history,
                "QUESTION": query
            }
            
            response = chain.invoke(str(input_dict))
            return response
        except Exception as e:
            return f"Error executing conversational chain: {str(e)}"

    # Qa Retrieval method 
    def QA_Retrieval(self, query, template, vector_store,k):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            _filter = LLMChainFilter.from_llm(self.model)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )
            setup_and_retrieval = RunnableParallel(
                {"CONTEXT": compression_retriever, "question": RunnablePassthrough()}
            )
            output_parser = StrOutputParser()
            rag_chain = (
                setup_and_retrieval
                | prompt
                | self.model
                | output_parser
            )
            response = rag_chain.invoke(query)
            return response
        except Exception as e:
            return f"Error executing retrieval chain: {str(e)}"


    # Conversational Retrieval method
    def Conversational_Retrieval(self, query, history, template, vector_store, k):
        try:
            prompt = ChatPromptTemplate.from_template(template)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
            _filter = LLMChainFilter.from_llm(self)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
                )
            setup_and_retrieval = RunnableParallel(
                {"CONTEXT": compression_retriever, "QUESTION": RunnablePassthrough(), "HISTORY": RunnablePassthrough()}
            )
            output_parser = StrOutputParser()
            rag_chain = (
                setup_and_retrieval
                | prompt
                | self.model
                | output_parser
            )
            input_dict = {"QUESTION": query, "HISTORY": history}
            response = rag_chain.invoke(str(input_dict))
            return response
        except Exception as e:
            return f"Error executing conversational retrieval chain: {str(e)}"



