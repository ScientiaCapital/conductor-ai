"""
Voice Call Tool
==============

Connects conductor-ai to VozLux for bilingual voice AI with
40ms Cartesia latency and Twilio integration.

Features:
- Bilingual (English/Spanish)
- 40ms voice synthesis latency
- Call scheduling
- Transcription and sentiment analysis
- Appointment booking integration
"""

import httpx
from typing import Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from ..base import BaseTool, ToolResult

# VozLux service URL
VOZLUX_URL = "http://localhost:8003"


class VoiceCallInput(BaseModel):
    """Input schema for voice calls."""
    
    action: Literal["schedule", "call_now", "check_status", "get_transcript"] = Field(
        ...,
        description="Action to perform"
    )
    phone_number: str = Field(..., description="Phone number in E.164 format (+1...)")
    contact_name: Optional[str] = Field(None, description="Contact name for personalization")
    language: Literal["en", "es"] = Field(default="en", description="Call language")
    call_type: Literal["qualification", "follow_up", "appointment", "custom"] = Field(
        default="qualification",
        description="Type of call script to use"
    )
    scheduled_time: Optional[datetime] = Field(None, description="Time to schedule call (for 'schedule' action)")
    custom_script: Optional[str] = Field(None, description="Custom call script (for 'custom' call_type)")
    call_id: Optional[str] = Field(None, description="Call ID (for 'check_status' or 'get_transcript')")
    max_duration_seconds: int = Field(default=300, ge=30, le=600, description="Maximum call duration")


class VoiceCallResult(BaseModel):
    """Output schema for voice calls."""
    
    call_id: str
    status: str = Field(..., description="scheduled, in_progress, completed, failed")
    phone_number: str
    language: str
    duration_seconds: Optional[float] = None
    transcript: Optional[str] = None
    sentiment: Optional[str] = None  # positive, neutral, negative
    key_points: list[str] = Field(default_factory=list)
    next_action: Optional[str] = None  # Follow-up recommendation
    appointment_scheduled: Optional[dict[str, Any]] = None
    cost_usd: float = Field(default=0.0, description="Call cost")


class VoiceCallTool(BaseTool):
    """
    Voice call tool connecting to VozLux bilingual voice AI.
    
    Features:
    - 40ms Cartesia voice synthesis latency
    - Bilingual support (English/Spanish)
    - Multiple call types (qualification, follow_up, appointment)
    - Automatic transcription
    - Sentiment analysis
    - Appointment booking integration
    
    Requires approval for outbound calls.
    """
    
    name: str = "voice_call"
    description: str = """Make or schedule bilingual voice calls for sales outreach.
    
    This tool:
    1. Schedules or immediately places outbound calls
    2. Uses AI voice synthesis (40ms latency)
    3. Transcribes and analyzes call sentiment
    4. Books appointments automatically
    5. Supports English and Spanish
    
    Best for: Lead qualification calls, appointment scheduling, follow-ups.
    Requires approval before placing outbound calls.
    """
    
    input_schema: type[BaseModel] = VoiceCallInput
    requires_approval: bool = True  # Important: requires human approval
    
    async def execute(self, input_data: VoiceCallInput) -> ToolResult:
        """Execute voice call action via VozLux."""
        
        try:
            endpoint_map = {
                "schedule": "/api/calls/schedule",
                "call_now": "/api/calls/start",
                "check_status": "/api/calls/status",
                "get_transcript": "/api/calls/transcript",
            }
            
            endpoint = endpoint_map[input_data.action]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "phone_number": input_data.phone_number,
                    "contact_name": input_data.contact_name,
                    "language": input_data.language,
                    "call_type": input_data.call_type,
                    "max_duration": input_data.max_duration_seconds,
                }
                
                if input_data.action == "schedule" and input_data.scheduled_time:
                    payload["scheduled_time"] = input_data.scheduled_time.isoformat()
                
                if input_data.call_type == "custom" and input_data.custom_script:
                    payload["script"] = input_data.custom_script
                
                if input_data.call_id:
                    payload["call_id"] = input_data.call_id
                
                response = await client.post(
                    f"{VOZLUX_URL}{endpoint}",
                    json=payload
                )
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"VozLux returned status {response.status_code}"
                    )
                
                result = response.json()
                
                call_result = VoiceCallResult(
                    call_id=result.get("call_id", ""),
                    status=result.get("status", "unknown"),
                    phone_number=input_data.phone_number,
                    language=input_data.language,
                    duration_seconds=result.get("duration"),
                    transcript=result.get("transcript"),
                    sentiment=result.get("sentiment"),
                    key_points=result.get("key_points", []),
                    next_action=result.get("next_action"),
                    appointment_scheduled=result.get("appointment"),
                    cost_usd=result.get("cost", 0.0),
                )
                
                return ToolResult(
                    success=True,
                    data=call_result.model_dump(),
                    metadata={
                        "action": input_data.action,
                        "call_type": input_data.call_type,
                    }
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Voice call failed: {str(e)}"
            )
    
    async def validate_input(self, input_data: VoiceCallInput) -> tuple[bool, Optional[str]]:
        """Validate voice call input."""
        
        # Validate phone number format (E.164)
        if not input_data.phone_number.startswith("+"):
            return False, "Phone number must be in E.164 format (starting with +)"
        
        if len(input_data.phone_number) < 10:
            return False, "Phone number too short"
        
        # Validate scheduled time is in the future
        if input_data.action == "schedule":
            if not input_data.scheduled_time:
                return False, "Scheduled time required for 'schedule' action"
            if input_data.scheduled_time < datetime.now():
                return False, "Scheduled time must be in the future"
        
        # Validate call_id for status/transcript actions
        if input_data.action in ["check_status", "get_transcript"]:
            if not input_data.call_id:
                return False, f"Call ID required for '{input_data.action}' action"
        
        return True, None
