from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os 

load_dotenv()

Model = os.getenv("MODEL")
Api_key = os.getenv("GEMINI_API_KEY")

llm = LLM(
    model=Model,
    api_key=Api_key,
    temperature=0.7,
)

@CrewBase
class LegalAi():
    """LegalAi crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def Petition_draft_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Petition_draft_agent'],
            verbose=True,
            memory=True,
            tools=[],
            allow_delegation=True
        )

    @task
    def Petition_draft_task(self) -> Task:
        return Task(
            config=self.tasks_config['Petition_draft_task'],
        )

    @agent
    def Petition_modifier_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Petition_modifier_agent'],
            verbose=True,
            memory=True,
            tools=[]
        )

    @task
    def Petition_modifier_task(self) -> Task:
        return Task(
            config=self.tasks_config['Petition_modifier_task'],
        )
        
    @crew
    def crew(self) -> Crew:
        """Creates the LegalAi crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,       
        )