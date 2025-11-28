"""
Academic Search Tool
===================

Connects conductor-ai to the perplexity-agent for 70x faster
research with Cerebras inference and multi-database academic search.

Databases:
- ArXiv (ML/AI papers)
- PubMed (building science, indoor air quality)
- IEEE (electrical/mechanical engineering)
- ACM (computer science)
- Semantic Scholar (cross-discipline)

Features:
- 7 research workflow templates
- Citation management (APA, MLA, IEEE, BibTeX)
- Conversation memory with session persistence
"""

import httpx
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from ..base import BaseTool, ToolResult

# Perplexity Agent URL
PERPLEXITY_AGENT_URL = "http://localhost:8002"


class AcademicSearchInput(BaseModel):
    """Input schema for academic search."""
    
    query: str = Field(..., min_length=3, description="Research query")
    databases: list[str] = Field(
        default=["arxiv", "semantic_scholar"],
        description="Databases to search: arxiv, pubmed, ieee, acm, semantic_scholar"
    )
    workflow: Literal[
        "literature_review",
        "systematic_review", 
        "exploratory",
        "comparative",
        "trend_analysis",
        "background",
        "citation_analysis"
    ] = Field(
        default="exploratory",
        description="Research workflow template"
    )
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results per database")
    citation_format: Literal["apa", "mla", "ieee", "chicago", "bibtex"] = Field(
        default="apa",
        description="Citation format for results"
    )
    date_range: Optional[str] = Field(
        None,
        description="Date range filter, e.g., '2020-2024'"
    )


class Paper(BaseModel):
    """Schema for a research paper."""
    
    title: str
    authors: list[str]
    abstract: str
    publication_date: Optional[str]
    source: str  # arxiv, pubmed, etc.
    url: str
    doi: Optional[str]
    citation: str  # Formatted citation
    relevance_score: float = Field(..., ge=0, le=1)


class AcademicSearchResult(BaseModel):
    """Output schema for academic search."""
    
    query: str
    total_results: int
    papers: list[Paper]
    synthesis: str = Field(..., description="AI-generated synthesis of findings")
    key_themes: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    citations_export: Optional[str] = Field(None, description="BibTeX/RIS export")
    inference_time_ms: float
    inference_cost_usd: float


class AcademicSearchTool(BaseTool):
    """
    Academic search tool using Cerebras-powered perplexity-agent.
    
    Features:
    - 70x faster inference with Cerebras
    - Multi-database search (ArXiv, PubMed, IEEE, ACM, Semantic Scholar)
    - 7 research workflow templates
    - Citation management (APA, MLA, IEEE, BibTeX export)
    - AI synthesis and theme extraction
    
    Cost: ~$0.001 per query via Cerebras
    """
    
    name: str = "academic_search"
    description: str = """Search academic databases for research papers and synthesize findings.
    
    This tool:
    1. Searches multiple academic databases simultaneously
    2. Ranks results by relevance
    3. Generates AI synthesis of key findings
    4. Identifies research themes and gaps
    5. Exports formatted citations
    
    Best for: Literature reviews, technical research, competitive analysis.
    Speed: 70x faster with Cerebras inference.
    """
    
    input_schema: type[BaseModel] = AcademicSearchInput
    requires_approval: bool = False
    
    async def execute(self, input_data: AcademicSearchInput) -> ToolResult:
        """Execute academic search via perplexity-agent."""
        
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{PERPLEXITY_AGENT_URL}/api/search",
                    json={
                        "query": input_data.query,
                        "databases": input_data.databases,
                        "workflow": input_data.workflow,
                        "max_results": input_data.max_results,
                        "citation_format": input_data.citation_format,
                        "date_range": input_data.date_range,
                    }
                )
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Academic search returned status {response.status_code}"
                    )
                
                result = response.json()
                
                # Parse papers
                papers = [
                    Paper(
                        title=p.get("title", ""),
                        authors=p.get("authors", []),
                        abstract=p.get("abstract", ""),
                        publication_date=p.get("date"),
                        source=p.get("source", ""),
                        url=p.get("url", ""),
                        doi=p.get("doi"),
                        citation=p.get("citation", ""),
                        relevance_score=p.get("relevance", 0.5),
                    )
                    for p in result.get("papers", [])
                ]
                
                search_result = AcademicSearchResult(
                    query=input_data.query,
                    total_results=len(papers),
                    papers=papers,
                    synthesis=result.get("synthesis", ""),
                    key_themes=result.get("themes", []),
                    research_gaps=result.get("gaps", []),
                    citations_export=result.get("bibtex"),
                    inference_time_ms=result.get("inference_time_ms", 0),
                    inference_cost_usd=result.get("cost", 0.001),
                )
                
                return ToolResult(
                    success=True,
                    data=search_result.model_dump(),
                    metadata={
                        "databases_searched": input_data.databases,
                        "workflow": input_data.workflow,
                    }
                )
                
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="Academic search timed out after 45 seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Academic search failed: {str(e)}"
            )
    
    async def validate_input(self, input_data: AcademicSearchInput) -> tuple[bool, Optional[str]]:
        """Validate academic search input."""
        
        valid_databases = ["arxiv", "pubmed", "ieee", "acm", "semantic_scholar"]
        for db in input_data.databases:
            if db not in valid_databases:
                return False, f"Invalid database: {db}. Valid: {', '.join(valid_databases)}"
        
        return True, None
