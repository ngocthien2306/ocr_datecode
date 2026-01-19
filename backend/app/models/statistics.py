"""
Statistics Models
Pydantic models for statistics API responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PeriodInfo(BaseModel):
    """Time period information"""
    start_date: datetime = Field(..., description="Start date of period (VN timezone)")
    end_date: datetime = Field(..., description="End date of period (VN timezone)")




class RecipeStats(BaseModel):
    """Statistics for a single recipe"""
    recipe_id: str = Field(..., description="Recipe identifier")
    recipe_name: str = Field(..., description="Recipe name")
    total: int = Field(..., description="Total inspections")
    pass_: int = Field(..., alias="pass", description="Number of PASS results")
    fail: int = Field(..., description="Number of FAIL results")
    pass_rate: float = Field(..., description="Pass rate percentage")


class SummaryStatisticsResponse(BaseModel):
    """Response model for summary statistics endpoint"""
    total: int = Field(..., description="Total inspections in period")
    pass_: int = Field(..., alias="pass", description="Total PASS results")
    fail: int = Field(..., description="Total FAIL results")
    pass_rate: float = Field(..., description="Overall pass rate percentage")
    period: PeriodInfo = Field(..., description="Query period information")
    by_recipe: List[RecipeStats] = Field(
        default_factory=list,
        description="Statistics breakdown by recipe"
    )


class TimeseriesRecipeData(BaseModel):
    """Recipe data for a single time point"""
    recipe_id: str = Field(..., description="Recipe identifier")
    recipe_name: str = Field(..., description="Recipe name")
    total: int = Field(..., description="Total inspections")
    pass_: int = Field(..., alias="pass", description="Number of PASS results")
    fail: int = Field(..., description="Number of FAIL results")
    pass_rate: float = Field(..., description="Pass rate percentage")


class TimeseriesDataPoint(BaseModel):
    """Single data point in timeseries"""
    timestamp: datetime = Field(..., description="Timestamp (VN timezone)")
    total: int = Field(..., description="Total inspections at this time point")
    pass_: int = Field(..., alias="pass", description="Number of PASS results")
    fail: int = Field(..., description="Number of FAIL results")
    pass_rate: float = Field(..., description="Pass rate percentage")
    by_recipe: List[TimeseriesRecipeData] = Field(
        default_factory=list,
        description="Breakdown by recipe"
    )


class TimeseriesStatisticsResponse(BaseModel):
    """Response model for timeseries statistics endpoint"""
    granularity: str = Field(..., description="Time granularity (hour or day)")
    period: PeriodInfo = Field(..., description="Query period information")
    data: List[TimeseriesDataPoint] = Field(
        default_factory=list,
        description="Timeseries data points"
    )
