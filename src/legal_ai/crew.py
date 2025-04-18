from crewai import Agent, Crew, Task, LLM, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectorySearchTool, FileReadTool
from dotenv import load_dotenv
import os 
from pathlib import Path
import json
load_dotenv()

Model = os.getenv("MODEL")
Api_key = os.getenv("GEMINI_API_KEY")
Embedding_model = os.getenv("EMBEDDING_MODEL")

import time
from threading import Lock
from crewai import LLM

llm = LLM(
    model=Model,
    api_key=Api_key,
    temperature=0.7,
    max_tokens=2000,
)



laws = DirectorySearchTool(
    directory='/home/amisha/Projects/legal_ai/legal_ai/knowledge/pdf_uscAll@119-4',
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini",
                api_key=Api_key
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model=Embedding_model,
                task_type="retrieval_document"            
            ),
        ),
    )
)

@CrewBase
class LegalAi():
    """LegalAi crew - defines agents, tasks, and crew configuration for legal document generation and modification."""

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
            tools=[laws],
            llm=llm
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
            tools=[laws],
            llm=llm

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
            tools=[],
            llm=llm
        )

    @agent
    def Summary_agent(self) -> Agent:
        """
        Defines the agent responsible for modifying existing petitions.

        Returns:
            Agent: An initialized summary generator agent.
        """
        return Agent(
            config=self.agents_config['Summary_agent'],  # Loads agent config from YAML
            tools=[laws],
            llm=llm
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
    def Summary_task(self) -> Task:
        """
        Defines the summary generator task for petitions.

        Returns:
            Task: A summary generator task object.
        """
        return Task(
            config=self.tasks_config['Summary_task'],  # Loads task config from YAML
        )  
           
    @crew
    def crew(self) -> Crew:
        """
        Assembles the LegalAi crew by bundling agents and tasks.

        Returns:
            Crew: A CrewAI Crew instance ready to run tasks.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            manager_llm=llm,
            process=Process.hierarchical,  
        )
