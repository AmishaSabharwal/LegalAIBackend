from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os 
from crewai.memory import ShortTermMemory, EntityMemory, LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage

load_dotenv()

Model = os.getenv("MODEL")
Api_key = os.getenv("GEMINI_API_KEY")
Open_api = os.getenv("OPEN_API_KEY")

llm = LLM(
    model=Model,
    api_key=Api_key,
    temperature=0.7
)

@CrewBase
class LegalAi():
    """LegalAi crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def draft_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['draft_agent'],
            verbose=True,
            llm=llm,
            memory=True
        )

    @task
    def draft_task(self) -> Task:
        return Task(
            config=self.tasks_config['draft_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LegalAi crew"""
        rag_storage = RAGStorage(
                        embedder_config={
                            "provider": "google",
                            "config": {
                                "model": 'models/gemini-embedding-exp-03-07',
                                "api_key": Api_key
                            }
                        },
                        type="short_term",
                        path="./tests/"
                    )
        
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            verbose=True,
            memory=True,
            long_term_memory = LongTermMemory(
                storage=LTMSQLiteStorage(
                    db_path="./tests/long_term/long_term_memory_storage.db"
                )
            ),
            short_term_memory = ShortTermMemory(storage = rag_storage),
            entity_memory=EntityMemory(storage=rag_storage)            
        )