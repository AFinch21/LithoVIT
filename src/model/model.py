from pydantic import BaseModel
from pydantic import UUID4
from typing import Optional, Union

# Define a Pydantic model for the response payload
class QueryRequest(BaseModel):
    message: str
    max_retries: int
    status: str
    
class EvaluationRequest(BaseModel):
    n_questions: int
    min_steps: int
    max_retries: int
    status: str