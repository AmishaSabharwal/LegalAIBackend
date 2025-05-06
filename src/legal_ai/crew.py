# Import required modules from CrewAI and standard Python libraries
from crewai import Agent, Crew, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os 
from src.legal_ai.tools.laws_tool import generate_rag_response

# Load environment variables from .env file
load_dotenv()
# Fetch API keys and model details from environment
Model = os.getenv("MODEL")
Api_key = os.getenv("GEMINI_API_KEY")
Embedding_model = os.getenv("EMBEDDING_MODEL")

# Initialize the language model with your chosen configuration
llm = LLM(
    model=Model,
    api_key=Api_key,
    temperature=0.3,
)

# Load the law-based RAG response tool
laws = generate_rag_response

@CrewBase
class LegalAi():
    """LegalAiBackend crew: This CrewBase defines all legal agents and their tasks."""

    # Paths to YAML configuration files for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    # ----------- AGENTS DEFINITION -----------

    @agent
    def drafting_agent(self) -> Agent:
        """
        Agent for drafting new legal petitions and contracts.
        Uses RAG tool to reference legal laws.
        """
        return Agent(
            config=self.agents_config['drafting_agent'],
            llm=llm,
            tools=[laws],
            max_iter=1,
        )

    @agent
    def modifier_agent(self) -> Agent:
        """
        Agent for modifying or updating existing legal documents.
        Suitable for contract revisions or argument updates.
        """
        return Agent(
            config=self.agents_config['modifier_agent'],
            llm=llm,
            tools=[laws],
            max_iter=1,
        )

    @agent
    def findings_agent(self) -> Agent:
        """
        Agent for comparing legal documents and extracting key findings.
        Useful for legal analysis and identifying inconsistencies.
        """
        return Agent(
            config=self.agents_config['findings_agent'],
            llm=llm,
            tools=[laws],
            max_iter=1,
        )

    @agent
    def summary_agent(self) -> Agent:
        """
        Agent for summarizing lengthy legal documents.
        Helps quickly understand petitions or contracts.
        """
        return Agent(
            config=self.agents_config['summary_agent'],
            llm=llm,
            tools=[laws],
            max_iter=1,
        ) 
        
    @agent
    def score_prediction(self) -> Agent:
        """
        Agent for predicting legal success probability or case outcomes.
        Analyzes inputs and provides a score or success likelihood.
        """
        return Agent(
            config=self.agents_config['score_prediction'],
            llm=llm,
            tools=[laws],
            max_iter=1,
        ) 
              
    # ----------- TASKS DEFINITION -----------

    @task
    def drafting_task(self) -> Task:
        """
        Task corresponding to drafting legal documents.
        Maps to the drafting agent.
        """
        return Task(
            config=self.tasks_config['drafting_task'],
        )
        
    @task
    def modifier_task(self) -> Task:
        """
        Task for modifying legal documents based on new input or corrections.
        """
        return Task(
            config=self.tasks_config['modifier_task'],
        )
       
    @task
    def findings_task(self) -> Task:
        """
        Task for performing legal comparisons and finding document similarities/differences.
        """
        return Task(
            config=self.tasks_config['findings_task'],
        )
             
    @task
    def summary_task(self) -> Task:
        """
        Task for summarizing lengthy legal documents into concise versions.
        """
        return Task(
            config=self.tasks_config['summary_task'],
        )
     
    @task
    def prediction_task(self) -> Task:
        """
        Task for predicting the outcome or success rate of legal cases.
        """
        return Task(
            config=self.tasks_config['prediction_task'],
        )
        
    # ----------- CREW GENERATION LOGIC -----------

    @crew
    def crew(self, task_name: str) -> Crew:
        """
        Dynamically generates a Crew based on the provided task name.
        Ensures only the required agent and task are used for efficiency.

        Args:
            task_name (str): The type of legal task (e.g., 'drafting', 'modifier').
            session_id (str): Unique session ID for tracking.

        Returns:
            Crew: A configured crew with the appropriate agent and task.
        """
        # Mapping task names to their respective agents and tasks
        task_agent_map = {
            'drafting': (self.drafting_agent(), self.drafting_task()),
            'modifier': (self.modifier_agent(), self.modifier_task()),
            'findings': (self.findings_agent(), self.findings_task()),
            'summary': (self.summary_agent(), self.summary_task()),
            'success': (self.score_prediction(), self.prediction_task())
        }

        # Fetch the right agent and task based on task_name
        agent, task = task_agent_map[task_name]

        # Create and return the Crew object with selected agent and task
        return Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            llm=llm,
        )
