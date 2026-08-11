from pydantic import BaseModel

class InsuranceInput(BaseModel):
    age : int
    bmi : float 
    children : int 
    is_female : int 
    is_smoker : int 
    region : str 