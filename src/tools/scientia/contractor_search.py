"""
Contractor Search Tool
=====================

Queries the dealer-scraper-mvp database of 8,277 qualified
MEP contractors with ICP scoring and multi-trade filtering.

Data Sources:
- State licensing databases (electrical, plumbing, HVAC)
- OEM dealer locators (Generac, Enphase, Mitsubishi, Daikin)
- Permit records and project history
- Company verification (revenue, employees, years in business)

ICP Scoring:
- Trade mix (self-perform vs subcontract)
- Revenue tier estimation
- License status and certifications
- Geographic coverage
"""

import httpx
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from ..base import BaseTool, ToolResult

# Dealer Scraper database service URL (or direct Supabase)
CONTRACTOR_DB_URL = "http://localhost:8005"


class ContractorSearchInput(BaseModel):
    """Input schema for contractor search."""
    
    query: Optional[str] = Field(None, description="Free text search query")
    trades: list[str] = Field(
        default_factory=list,
        description="Filter by trades: HVAC, Electrical, Plumbing, Solar, Roofing, MEP"
    )
    states: list[str] = Field(
        default_factory=list,
        description="Filter by state codes (e.g., ['TX', 'CA', 'FL'])"
    )
    min_icp_score: float = Field(default=0, ge=0, le=100, description="Minimum ICP score")
    certifications: list[str] = Field(
        default_factory=list,
        description="Required certifications: Generac, Enphase, Mitsubishi, Daikin, etc."
    )
    revenue_tier: Optional[Literal["small", "medium", "large", "enterprise"]] = Field(
        None,
        description="Revenue tier filter"
    )
    self_perform: Optional[bool] = Field(
        None,
        description="Filter for self-performing contractors only"
    )
    commercial: Optional[bool] = Field(
        None,
        description="Filter for commercial project capability"
    )
    license_active: bool = Field(default=True, description="Require active license")
    limit: int = Field(default=25, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class Contractor(BaseModel):
    """Schema for a contractor record."""
    
    id: str
    company_name: str
    website: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: str
    zip_code: Optional[str]
    trades: list[str]
    certifications: list[str]
    licenses: list[dict[str, Any]]  # License numbers and types
    icp_score: float
    revenue_estimate: Optional[str]
    employee_estimate: Optional[str]
    years_in_business: Optional[int]
    self_perform: bool
    commercial_capable: bool
    residential_capable: bool
    data_freshness: str  # When data was last verified


class ContractorSearchResult(BaseModel):
    """Output schema for contractor search."""
    
    total_count: int
    returned_count: int
    contractors: list[Contractor]
    filters_applied: dict[str, Any]
    offset: int
    has_more: bool


class ContractorSearchTool(BaseTool):
    """
    Contractor database search tool via dealer-scraper-mvp.
    
    Features:
    - 8,277 qualified MEP contractors
    - ICP scoring with multi-factor analysis
    - Multi-state license verification
    - OEM certification tracking
    - Trade mix analysis
    - Revenue estimation from permits
    
    Data sources:
    - State licensing boards
    - OEM dealer locators (15+ OEMs)
    - Permit databases
    - Company registrations
    """
    
    name: str = "contractor_search"
    description: str = """Search the proprietary database of 8,277 qualified MEP contractors.
    
    This tool:
    1. Searches contractors by trade, location, certifications
    2. Filters by ICP score and revenue tier
    3. Verifies license status and certifications
    4. Returns ranked results with full contact info
    
    Data includes: HVAC, Electrical, Plumbing, Solar, Roofing contractors
    with Generac, Enphase, Mitsubishi, Daikin certifications.
    """
    
    input_schema: type[BaseModel] = ContractorSearchInput
    requires_approval: bool = False
    
    async def execute(self, input_data: ContractorSearchInput) -> ToolResult:
        """Execute contractor search."""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{CONTRACTOR_DB_URL}/api/contractors/search",
                    json={
                        "query": input_data.query,
                        "trades": input_data.trades,
                        "states": input_data.states,
                        "min_icp_score": input_data.min_icp_score,
                        "certifications": input_data.certifications,
                        "revenue_tier": input_data.revenue_tier,
                        "self_perform": input_data.self_perform,
                        "commercial": input_data.commercial,
                        "license_active": input_data.license_active,
                        "limit": input_data.limit,
                        "offset": input_data.offset,
                    }
                )
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Contractor search returned status {response.status_code}"
                    )
                
                result = response.json()
                
                contractors = [
                    Contractor(
                        id=c.get("id", ""),
                        company_name=c.get("company_name", ""),
                        website=c.get("website"),
                        phone=c.get("phone"),
                        email=c.get("email"),
                        address=c.get("address"),
                        city=c.get("city"),
                        state=c.get("state", ""),
                        zip_code=c.get("zip_code"),
                        trades=c.get("trades", []),
                        certifications=c.get("certifications", []),
                        licenses=c.get("licenses", []),
                        icp_score=c.get("icp_score", 0),
                        revenue_estimate=c.get("revenue_estimate"),
                        employee_estimate=c.get("employee_estimate"),
                        years_in_business=c.get("years_in_business"),
                        self_perform=c.get("self_perform", False),
                        commercial_capable=c.get("commercial_capable", False),
                        residential_capable=c.get("residential_capable", True),
                        data_freshness=c.get("data_freshness", "unknown"),
                    )
                    for c in result.get("contractors", [])
                ]
                
                search_result = ContractorSearchResult(
                    total_count=result.get("total_count", 0),
                    returned_count=len(contractors),
                    contractors=contractors,
                    filters_applied={
                        "trades": input_data.trades,
                        "states": input_data.states,
                        "min_icp_score": input_data.min_icp_score,
                        "certifications": input_data.certifications,
                    },
                    offset=input_data.offset,
                    has_more=result.get("has_more", False),
                )
                
                return ToolResult(
                    success=True,
                    data=search_result.model_dump(),
                    metadata={
                        "total_available": result.get("total_count", 0),
                    }
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Contractor search failed: {str(e)}"
            )
    
    async def validate_input(self, input_data: ContractorSearchInput) -> tuple[bool, Optional[str]]:
        """Validate contractor search input."""
        
        valid_trades = ["HVAC", "Electrical", "Plumbing", "Solar", "Roofing", "MEP"]
        for trade in input_data.trades:
            if trade not in valid_trades:
                return False, f"Invalid trade: {trade}. Valid: {', '.join(valid_trades)}"
        
        # Validate state codes (2 letters)
        for state in input_data.states:
            if len(state) != 2 or not state.isalpha():
                return False, f"Invalid state code: {state}. Use 2-letter codes (TX, CA, etc.)"
        
        return True, None
