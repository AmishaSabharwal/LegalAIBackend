from crewai import Agent, Crew, Task, LLM, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os 
from src.legal_ai.tools.laws_tool import generate_rag_response

load_dotenv()
Model = os.getenv("MODEL")
Api_key = os.getenv("GEMINI_API_KEY")

llm = LLM(
    model=Model,
    api_key=Api_key,
    temperature=0.3,
)

laws = generate_rag_response

@CrewBase
class LegalAi():
    """LegalAiBackend crew"""

    # Paths to YAML config files for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def drafting_agent(self) -> Agent:
        """
        Defines the agent responsible for drafting legal petitions and contracts.

        Returns:
            Agent: An initialized drafting agent.
        """
        return Agent(
            config=self.agents_config['drafting_agent'],  # Loads agent config from YAML
            llm=llm,
            tools=[laws],
            max_iter=1,
        )

    @agent
    def modifier_agent(self) -> Agent:
        """
        Defines the agent responsible for modifying existing petitions and contracts.

        Returns:
            Agent: An initialized modifying agent.
        """
        return Agent(
            config=self.agents_config['modifier_agent'],  # Loads agent config from YAML
            llm=llm,
            max_iter=1,
        )

    @agent
    def findings_agent(self) -> Agent:
        """
        Defines the agent responsible for modifying existing petitions.

        Returns:
            Agent: An initialized summary generator agent.
        """
        return Agent(
            config=self.agents_config['findings_agent'],  # Loads agent config from YAML
            llm=llm,
            max_iter=1,
        )

    @agent
    def summary_agent(self) -> Agent:
        """
        Defines the agent responsible for modifying existing petitions.

        Returns:
            Agent: An initialized summary generator agent.
        """
        return Agent(
            config=self.agents_config['summary_agent'],  # Loads agent config from YAML
            llm=llm,
            tools=[laws],
            max_iter=1,
        ) 
        
    @agent
    def score_prediction(self) -> Agent:
        """
        Defines the agent responsible for modifying existing petitions.

        Returns:
            Agent: An initialized summary generator agent.
        """
        return Agent(
            config=self.agents_config['score_prediction'],  # Loads agent config from YAML
            llm=llm,
            max_iter=1,
        ) 
              
    @task
    def drafting_task(self) -> Task:
        """
        Defines the drafting task for petitions and contracts.

        Returns:
            Task: A drafting task object.
        """
        return Task(
            config=self.tasks_config['drafting_task'],  # Loads task config from YAML
        )
        
    @task
    def modifier_task(self) -> Task:
        """
        Defines the modification task for reviewing and editing petitions and contracts.

        Returns:
            Task: A modifier task object.
        """
        return Task(
            config=self.tasks_config['modifier_task'],  # Loads task config from YAML
        )
       
    @task
    def findings_task(self) -> Task:
        """
        Defines the summary generator task for petitions.

        Returns:
            Task: A summary generator task object.
        """
        return Task(
            config=self.tasks_config['findings_task'],  # Loads task config from YAML
        )
             
    @task
    def summary_task(self) -> Task:
        """
        Defines the summary generator task for petitions.

        Returns:
            Task: A summary generator task object.
        """
        return Task(
            config=self.tasks_config['summary_task'],  # Loads task config from YAML
        )
     
    @task
    def prediction_task(self) -> Task:
        """
        Defines the summary generator task for petitions.

        Returns:
            Task: A summary generator task object.
        """
        return Task(
            config=self.tasks_config['prediction_task'],  # Loads task config from YAML
        )
        
    @crew
    def crew(self, task_name: str) -> Crew:
        """
        Dynamically creates a crew with only the agent and task relevant to the query.
        """
        task_agent_map = {
            'drafting': (self.drafting_agent(), self.drafting_task()),
            'modifier': (self.modifier_agent(), self.modifier_task()),
            'findings': (self.findings_agent(), self.findings_task()),
            'summary': (self.summary_agent(), self.summary_task()),
            'success': (self.score_prediction(), self.prediction_task())
        }

        agent, task = task_agent_map[task_name]

        return Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            llm=llm,
        )
