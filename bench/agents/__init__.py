from .long_context import LongContextAgent
from .rag_naive import NaiveRAGAgent
from .rag_policy import PolicyRAGAgent
from .a_mem import AMemAgent
from .mem0 import Mem0Agent
from .remem import ReMemAgent
from .example_agent import ExampleAgent

AGENT_REGISTRY = {
    "long_context": LongContextAgent,
    "rag_naive": NaiveRAGAgent,
    "rag_policy": PolicyRAGAgent,
    "a_mem": AMemAgent,
    "mem0": Mem0Agent,
    "remem": ReMemAgent,
    "example": ExampleAgent,
}