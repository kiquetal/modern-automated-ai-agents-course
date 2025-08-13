"""
Wikipedia Question Answering with CrewAI and LangChain
This script creates agents that can read content from Wikipedia and answer questions about it.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Make sure to set your OPENAI_API_KEY in a .env file or as an environment variable
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o")

class WikipediaResearchCrew:
    def __init__(self, wiki_url, questions):
        """
        Initialize the Wikipedia research crew.
        
        Args:
            wiki_url (str): URL of the Wikipedia page to analyze
            questions (list): List of questions to answer about the content
        """
        self.wiki_url = wiki_url
        self.questions = questions
        self.setup_crew()
    
    def fetch_wikipedia_content(self):
        """Load and split the content from the Wikipedia URL"""
        # Load content from URL
        loader = WebBaseLoader(self.wiki_url)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    
    def setup_crew(self):
        """Set up the crew with researcher and analyst agents"""
        # Create a researcher agent
        self.researcher = Agent(
            role="Wikipedia Researcher",
            goal="Extract and organize key information from Wikipedia articles",
            backstory="You are an expert at reading and extracting important information from Wikipedia pages",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        # Create an analyst agent
        self.analyst = Agent(
            role="Content Analyst",
            goal="Analyze content and provide insightful answers to questions",
            backstory="You are an expert analyst who can understand complex information and provide clear, accurate answers",
            verbose=True,
            allow_delegation=True,
            llm=llm
        )
        
        # Define tasks
        self.research_task = Task(
            description=f"Read and extract key information from the Wikipedia page at {self.wiki_url}",
            agent=self.researcher,
            expected_output="A comprehensive summary of the Wikipedia page content"
        )
        
        questions_text = "\n".join([f"- {question}" for question in self.questions])
        self.analysis_task = Task(
            description=f"Based on the research, answer the following questions:\n{questions_text}",
            agent=self.analyst,
            expected_output="Detailed answers to each question based on the Wikipedia content"
        )
        
        # Create the crew
        self.crew = Crew(
            agents=[self.researcher, self.analyst],
            tasks=[self.research_task, self.analysis_task],
            verbose=2
        )
    
    def run(self):
        """Run the crew to answer the questions"""
        # First, fetch the content
        documents = self.fetch_wikipedia_content()
        
        # Add the content to the context
        context = f"Here is the content from Wikipedia: \n\n"
        for doc in documents:
            context += doc.page_content + "\n\n"
        
        # Update the research task with the content
        self.research_task.context = context
        
        # Run the crew
        result = self.crew.kickoff()
        return result

# Example usage
if __name__ == "__main__":
    # Example Wikipedia URL about artificial intelligence
    wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    # Example questions to answer
    questions = [
        "What is the history of artificial intelligence?",
        "What are the main approaches to AI?",
        "What ethical concerns are associated with AI development?",
        "How is AI being used in industry today?"
    ]
    
    # Create and run the crew
    wikipedia_crew = WikipediaResearchCrew(wiki_url, questions)
    result = wikipedia_crew.run()
    
    print("\n\n=== FINAL RESULTS ===\n\n")
    print(result)
