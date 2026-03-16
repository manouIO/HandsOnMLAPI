from pydantic import BaseModel
from typing import Literal


class HousePredictionInput(BaseModel):
    area_sqft: float
    num_toilets: int
    house_type: Literal["apartment", "villa", "condo"]
    dist_to_center_km: float  
    has_garden: bool         

class PredictionOutput(BaseModel):
    house_type: str
    estimated_price: str
    #currency: str = "USD"