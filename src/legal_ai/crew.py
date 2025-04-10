from crewai import Agent, Crew, Task, LLM, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DirectorySearchTool, DirectoryReadTool
from dotenv import load_dotenv
import os 

load_dotenv()

Model = os.getenv("MODEL")
Api_key = os.getenv("GEMINI_API_KEY")
Embedding_model= os.getenv("EMBEDDING_MODEL")

llm = LLM(
    model=Model,
    api_key=Api_key,
    temperature=0.7,
    max_tokens=1000,
    timeout=120,
    embedding_model=Embedding_model
)

# laws = DirectorySearchTool(
#     directory='./knowledge/pdf_uscAll@119-4',  # path to folder containing PDFs
#     config=dict(
#         llm=dict(
#             provider="google",
#             config=dict(
#                 model="gemini",
#                 api_key=Api_key
#             ),
#         ),
#         embedder=dict(
#             provider="google",
#             config=dict(
#                 model=Embedding_model,
#                 task_type="retrieval_document"            
#             ),
#         ),
#     )
# )
# laws = DirectorySearchTool(
#     directory='/home/amisha/Projects/legal_ai/legal_ai/knowledge/pdf_uscAll@119-4',
#     config={
#         "llm": {
#             "provider": "google",
#             "config": {
#                 "model": Model,
#                 "api_key": Api_key
#             },
#         },
#         "embedder": {
#             "provider": "google",
#             "config": {
#                 "model": Embedding_model,  # Same here since it's used for embeddings
#                 "task_type": "retrieval_document",
#             },
#         },
#     }
# )

laws = DirectoryReadTool(directory='/path/to/your/directory')

@CrewBase
class LegalAi():
    """LegalAi crew - defines agents, tasks, and crew configuration for legal document generation and modification."""

    # Paths to YAML config files for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def drafting_agent(self) -> Agent:
        """
        Defines the agent responsible for drafting legal petitions.

        Returns:
            Agent: An initialized drafting agent.
        """
        return Agent(
            config=self.agents_config['drafting_agent'],  # Loads agent config from YAML
            verbose=True,  # Enables detailed logging
            memory=True,  # Enables conversation memory
            llm=llm,
            tools=[laws]
        )

    @task
    def drafting_task(self) -> Task:
        """
        Defines the drafting task for petitions.

        Returns:
            Task: A drafting task object.
        """
        return Task(
            config=self.tasks_config['drafting_task'],  # Loads task config from YAML
        )

    @agent
    def modifier_agent(self) -> Agent:
        """
        Defines the agent responsible for modifying existing petitions.

        Returns:
            Agent: An initialized modifying agent.
        """
        return Agent(
            config=self.agents_config['modifier_agent'],  # Loads agent config from YAML
            verbose=True,
            memory=True,
            llm=llm,
            tools=[laws]
        )

    @task
    def modifier_task(self) -> Task:
        """
        Defines the modification task for reviewing and editing petitions.

        Returns:
            Task: A modifier task object.
        """
        return Task(
            config=self.tasks_config['modifier_task'],  # Loads task config from YAML
        )
        
    @crew
    def crew(self) -> Crew:
        """
        Assembles the LegalAi crew by bundling agents and tasks.

        Returns:
            Crew: A CrewAI Crew instance ready to run tasks.
        """
        return Crew(
            agents=self.agents,  # Collects all @agent methods
            tasks=self.tasks,    # Collects all @task methods
            verbose=True,        # Enables verbose output during execution
        )
