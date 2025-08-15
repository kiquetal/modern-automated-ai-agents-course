#### Course: Modern Automated AI agents: Building agentic AI to Perfom Complex Tasks


##### Lesson1: Introduction to AI Agents

- Autonomy, decision-making, adaptation (learn and improve over with time feedback)

Agent: Performs specific taks and makes decisiones based on its environemnt
Large Language Model: focuses on understanding and generating human-like text.


Productivity and Efficiency: automate repetitive tasks, freeuing up human resources for more complex activities.
Handle dynamic and real-time environments like finance or customer service.


- Leading AI Agent framework: LangChain: designed for LLMs, supports agent workflow for NLP and decision-making tasks.
- CrewAI: focuses on collabolartieve, role-bases AI agents that work in teams to tackle complex tasks

 





##### Lesson2: Under the Hood of AI Agents







##### Lesson3:

#### Course: Modern Automated AI agents: Building agentic AI to Perfom Complex Tasks


##### Lesson1: Introduction to AI Agents

- Autonomy, decision-making, adaptation (learn and improve over with time feedback)

Agent: Performs specific taks and makes decisiones based on its environemnt
Large Language Model: focuses on understanding and generating human-like text.


Productivity and Efficiency: automate repetitive tasks, freeuing up human resources for more complex activities.
Handle dynamic and real-time environments like finance or customer service.


- Leading AI Agent framework: LangChain: designed for LLMs, supports agent workflow for NLP and decision-making tasks.
- CrewAI: focuses on collabolartieve, role-bases AI agents that work in teams to tackle complex tasks

 





##### Lesson2: Under the Hood of AI Agents







##### Lesson3:



---

## Use the App (Streamlit)

This project includes a Streamlit app that lets you run a Wikipedia + custom data QA workflow using CrewAI and LangChain.

### Prerequisites
- Python 3.10+ recommended
- An OpenAI API key

### Installation
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   - pip install -r requirements.txt

### OpenAI API Key
You can provide your API key in either of the following ways:
- Create a .env file in the project root with:
  - OPENAI_API_KEY=your_key_here
- Or enter your key directly in the Streamlit app sidebar when running.

### Run the App
- From the project root, run:
  - streamlit run app.py

This will open a browser window for the app. If it doesn’t open automatically, visit the URL shown in your terminal (usually http://localhost:8501).

### How to Use
In the app UI:
- Enter your OpenAI API key in the sidebar (unless it’s already set by .env).
- Choose a model (e.g., gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-3.5-turbo).
- Adjust chunk size and chunk overlap if needed.
- Provide one or more Wikipedia URLs (one per line).
- Optionally add extra RAG URLs (any web pages to include as additional sources).
- Optionally paste custom text to include in the context.
- Enter your questions (one per line).
- Click “Run QA” to execute the agents and view the results.

### Notes & Troubleshooting
- If you don’t provide at least one source URL or some custom text, the app will ask for more context.
- If you don’t provide any questions, the app will ask you to add at least one.
- Some sites may block scraping or require different headers; per-URL errors are shown as warnings in the UI, but other URLs will still load.
- wikipedia_qa.py is a stand‑alone script that enforces OPENAI_API_KEY at import time. It’s recommended to use the Streamlit app (app.py) for interactive usage. If you want to run wikipedia_qa.py directly, ensure OPENAI_API_KEY is set in your environment first.
