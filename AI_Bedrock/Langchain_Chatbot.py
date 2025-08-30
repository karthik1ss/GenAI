#!/usr/bin/env python3

import boto3
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from IPython.display import Markdown, display

def create_langchain_chatbot():
    """Create and configure a LangChain chatbot using Bedrock"""
    
    # Create AWS session
    session = boto3.Session()
    
    # Use a model that we know works (Titan Text Large)
    llm = BedrockLLM(
        model_id="amazon.titan-tg1-large",  # Fixed: Use working model
        model_kwargs={
            "temperature": 0.0, 
            "maxTokenCount": 500,
            "topP": 1.0
        },
        region_name="us-west-2"  # Fixed: Specify region explicitly
    )
    
    # Define conversation template
    template = """The following is a friendly conversation between a human and an AI. \
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
    
    # Create prompt template with correct template variable
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=template  # Fixed: Add template parameter
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="AI", human_prefix="Human"),
        prompt=prompt_template  # Fixed: Use 'prompt' instead of 'prompt_template'
    )
    
    return conversation

def test_chatbot():
    """Test the chatbot with a sample conversation"""
    try:
        print("🤖 Creating LangChain Bedrock Chatbot...")
        conversation = create_langchain_chatbot()
        
        print("✅ Chatbot created successfully!")
        print("\n" + "="*60)
        print("Testing conversation...")
        print("="*60)
        
        # Test conversation
        test_questions = [
            "Hello! What can you help me with?",
            "Can you name a few real-life applications of natural language processing?",
            "What did I ask you about in my previous question?",
            "Hello! Are you familiar with Machine Learning?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🗣️  Question {i}: {question}")
            print("🤖 Response:")
            response = conversation.predict(input=question)
            print(response)
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have langchain-aws installed: pip install langchain-aws")
        print("2. Check your AWS credentials are configured")
        print("3. Ensure you have access to the Titan model")

if __name__ == "__main__":
    test_chatbot()








